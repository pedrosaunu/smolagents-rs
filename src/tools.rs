use anyhow::Result;
use htmd::HtmlToMarkdown;
use reqwest::Url;
use schemars::gen::SchemaSettings;
use schemars::schema::RootSchema;
use schemars::JsonSchema;
use scraper::Selector;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::local_python_interpreter::evaluate_python_code;
use crate::models::openai::FunctionCall;

pub trait Parameters: DeserializeOwned + JsonSchema {}
pub trait Tool: Debug {
    type Params: Parameters;
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn forward(&self, arguments: Self::Params) -> Result<String>;
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Serialize, Debug)]
pub struct ToolInfo {
    #[serde(rename = "type")]
    tool_type: ToolType,
    pub function: ToolFunctionInfo,
}

#[derive(Serialize, Debug)]
pub struct ToolFunctionInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: RootSchema,
}

impl ToolInfo {
    pub fn new<P: Parameters, T: AnyTool>(tool: &T) -> Self {
        let mut settings = SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let generator = settings.into_generator();

        let parameters = generator.into_root_schema_for::<P>();

        Self {
            tool_type: ToolType::Function,
            function: ToolFunctionInfo {
                name: tool.name(),
                description: tool.description(),
                parameters,
            },
        }
    }

    pub fn get_parameter_names(&self) -> Vec<String> {
        if let Some(schema) = &self.function.parameters.schema.object {
            return schema.properties.keys().cloned().collect();
        }
        Vec::new()
    }
}

pub fn get_json_schema(tool: &ToolInfo) -> serde_json::Value {
    json!(tool)
}

pub trait ToolGroup: Debug {
    fn call(&self, arguments: &FunctionCall) -> Result<String>;
    fn tool_info(&self) -> Vec<ToolInfo>;
}

impl ToolGroup for Vec<Box<dyn AnyTool>> {
    fn call(&self, arguments: &FunctionCall) -> Result<String> {
        let tool = self.iter().find(|tool| tool.name() == arguments.name);
        if let Some(tool) = tool {
            let p = arguments.arguments.clone();
            return tool.forward_json(p);
        }
        Err(anyhow::anyhow!("Tool not found"))
    }
    fn tool_info(&self) -> Vec<ToolInfo> {
        self.iter().map(|tool| tool.tool_info()).collect()
    }
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "BaseParams")]
pub struct BaseParams {
    #[schemars(description = "The name of the tool")]
    _name: String,
}

impl<P: DeserializeOwned + JsonSchema> Parameters for P where P: JsonSchema {}

#[derive(Debug, Serialize, Default, Clone)]
pub struct BaseTool {
    pub name: &'static str,
    pub description: &'static str,
}

impl Tool for BaseTool {
    type Params = serde_json::Value;
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }
    fn forward(&self, _arguments: serde_json::Value) -> Result<String> {
        Ok("Not implemented".to_string())
    }
}
#[derive(Debug, Serialize, Default, Clone)]
pub struct VisitWebsiteTool {
    pub tool: BaseTool,
}

impl VisitWebsiteTool {
    pub fn new() -> Self {
        VisitWebsiteTool {
            tool: BaseTool {
                name: "visit_website",
                description: "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages",
            },
        }
    }

    pub fn forward(&self, url: &str) -> String {
        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        let url = match Url::parse(url) {
            Ok(url) => url,
            Err(_) => Url::parse(&format!("https://{}", url)).unwrap(),
        };

        let response = client.get(url).send();

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text() {
                        Ok(text) => {
                            let converter = HtmlToMarkdown::builder()
                                .skip_tags(vec!["script", "style"])
                                .build();
                            converter.convert(&text).unwrap()
                        }
                        Err(_) => "Failed to read response text".to_string(),
                    }
                } else if resp.status().as_u16() == 999 {
                    "The website appears to be blocking automated access. Try visiting the URL directly in your browser.".to_string()
                } else {
                    format!(
                        "Failed to fetch the webpage: HTTP {} - {}",
                        resp.status(),
                        resp.status().canonical_reason().unwrap_or("Unknown Error")
                    )
                }
            }
            Err(e) => format!("Failed to make the request: {}", e),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "VisitWebsiteToolParams")]
pub struct VisitWebsiteToolParams {
    #[schemars(description = "The url of the website to visit")]
    url: String,
}

impl Tool for VisitWebsiteTool {
    type Params = VisitWebsiteToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }

    fn description(&self) -> &'static str {
        self.tool.description
    }

    fn forward(&self, arguments: VisitWebsiteToolParams) -> Result<String> {
        let url = arguments.url;
        Ok(self.forward(&url))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(title = "FinalAnswerToolParams")]
pub struct FinalAnswerToolParams {
    #[schemars(description = "The final answer to the problem")]
    answer: String,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct FinalAnswerTool {
    pub tool: BaseTool,
}

impl FinalAnswerTool {
    pub fn new() -> Self {
        FinalAnswerTool {
            tool: BaseTool {
                name: "final_answer",
                description: "Provides a final answer to the given problem.",
            },
        }
    }
}

impl Tool for FinalAnswerTool {
    type Params = FinalAnswerToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }

    fn forward(&self, arguments: FinalAnswerToolParams) -> Result<String> {
        Ok(arguments.answer)
    }
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "GoogleSearchToolParams")]
pub struct GoogleSearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
    #[schemars(description = "Optionally restrict results to a certain year")]
    filter_year: Option<String>,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct GoogleSearchTool {
    pub tool: BaseTool,
    pub api_key: String,
}

impl GoogleSearchTool {
    pub fn new(api_key: Option<String>) -> Self {
        let api_key = api_key.unwrap_or(std::env::var("SERPAPI_API_KEY").unwrap());

        GoogleSearchTool {
            tool: BaseTool {
                name: "google_search",
                description: "Performs a google web search for your query then returns a string of the top search results.",
            },
            api_key,
        }
    }

    fn forward(&self, query: &str, filter_year: Option<&str>) -> String {
        let params = {
            let mut params = json!({
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "google_domain": "google.com",
            });

            if let Some(year) = filter_year {
                params["tbs"] = json!(format!("cdr:1,cd_min:01/01/{},cd_max:12/31/{}", year, year));
            }

            params
        };

        let client = reqwest::blocking::Client::new();
        let response = client
            .get("https://serpapi.com/search.json")
            .query(&params)
            .send();
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let results: serde_json::Value = resp.json().unwrap();
                    if results.get("organic_results").is_none() {
                        if filter_year.is_some() {
                            return format!("'organic_results' key not found for query: '{}' with filtering on year={}. Use a less restrictive query or do not filter on year.", query, filter_year.unwrap());
                        } else {
                            return format!("'organic_results' key not found for query: '{}'. Use a less restrictive query.", query);
                        }
                    }

                    let organic_results =
                        results.get("organic_results").unwrap().as_array().unwrap();
                    if organic_results.is_empty() {
                        let _ = if filter_year.is_some() {
                            format!(" with filter year={}", filter_year.unwrap())
                        } else {
                            "".to_string()
                        };
                        return format!("No results found for '{}'. Try with a more general query, or remove the year filter.", query);
                    }

                    let mut web_snippets = Vec::new();
                    for (idx, page) in organic_results.iter().enumerate() {
                        let date_published = page.get("date").map_or("".to_string(), |d| {
                            format!("\nDate published: {}", d.as_str().unwrap_or(""))
                        });
                        let source = page.get("source").map_or("".to_string(), |s| {
                            format!("\nSource: {}", s.as_str().unwrap_or(""))
                        });
                        let snippet = page.get("snippet").map_or("".to_string(), |s| {
                            format!("\n{}", s.as_str().unwrap_or(""))
                        });

                        let redacted_version = format!(
                            "{}. [{}]({}){}{}\n{}",
                            idx,
                            page.get("title").unwrap().as_str().unwrap(),
                            page.get("link").unwrap().as_str().unwrap(),
                            date_published,
                            source,
                            snippet
                        );
                        let redacted_version =
                            redacted_version.replace("Your browser can't play this video.", "");
                        web_snippets.push(redacted_version);
                    }

                    format!("## Search Results\n{}", web_snippets.join("\n\n"))
                } else {
                    format!(
                        "Failed to fetch search results: HTTP {}, Error: {}",
                        resp.status(),
                        resp.text().unwrap()
                    )
                }
            }
            Err(e) => format!("Failed to make the request: {}", e),
        }
    }
}

impl Tool for GoogleSearchTool {
    type Params = GoogleSearchToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }

    fn forward(&self, arguments: GoogleSearchToolParams) -> Result<String> {
        let query = arguments.query;
        let filter_year = arguments.filter_year;
        Ok(self.forward(&query, filter_year.as_deref()))
    }
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "DuckDuckGoSearchToolParams")]
pub struct DuckDuckGoSearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
}

#[derive(Debug, Serialize, Default)]
pub struct SearchResult {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct DuckDuckGoSearchTool {
    pub tool: BaseTool,
}

impl DuckDuckGoSearchTool {
    pub fn new() -> Self {
        DuckDuckGoSearchTool {
            tool: BaseTool {
                name: "duckduckgo_search",
                description: "Performs a duckduckgo web search for your query then returns a string of the top search results.",
            },
        }
    }

    pub fn forward(&self, query: &str) -> Result<Vec<SearchResult>> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; MyRustTool/1.0)")
            .build()?;
        let response = client
            .get(format!("https://html.duckduckgo.com/html/?q={}", query))
            .send()?;
        let html = response.text().unwrap();
        let document = scraper::Html::parse_document(&html);
        let result_selector = Selector::parse(".result")
            .map_err(|e| anyhow::anyhow!("Failed to parse result selector: {}", e))?;
        let title_selector = Selector::parse(".result__title a")
            .map_err(|e| anyhow::anyhow!("Failed to parse title selector: {}", e))?;
        let snippet_selector = Selector::parse(".result__snippet")
            .map_err(|e| anyhow::anyhow!("Failed to parse snippet selector: {}", e))?;
        let mut results = Vec::new();

        for result in document.select(&result_selector) {
            let title_element = result.select(&title_selector).next();
            let snippet_element = result.select(&snippet_selector).next();
            if let (Some(title), Some(snippet)) = (title_element, snippet_element) {
                let title_text = title.text().collect::<String>().trim().to_string();
                let snippet_text = snippet.text().collect::<String>().trim().to_string();
                let url = title
                    .value()
                    .attr("href")
                    .and_then(|href| {
                        // Parse and clean the URL
                        if href.starts_with("//") {
                            // Handle protocol-relative URLs
                            Some(format!("https:{}", href))
                        } else if href.starts_with('/') {
                            // Handle relative URLs
                            Some(format!("https://duckduckgo.com{}", href))
                        } else if let Ok(parsed_url) = Url::parse(href) {
                            // Handle absolute URLs
                            Some(parsed_url.to_string())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                if !title_text.is_empty() && !url.is_empty() {
                    results.push(SearchResult {
                        title: title_text,
                        snippet: snippet_text,
                        url,
                    });
                }
            }
        }
        Ok(results)
    }
}

impl Tool for DuckDuckGoSearchTool {
    type Params = DuckDuckGoSearchToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }
    fn forward(&self, arguments: DuckDuckGoSearchToolParams) -> Result<String> {
        let query = arguments.query;
        let results = self.forward(&query)?;
        let json_string = serde_json::to_string_pretty(&results)?;
        Ok(json_string)
    }
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "PythonInterpreterToolParams")]
pub struct PythonInterpreterToolParams {
    #[schemars(
        description = "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, 
        else you will get an error. 
        This code can only import the following python libraries: 
        collections, datetime, itertools, math, queue, random, re, stat, statistics, time, unicodedata"
    )]
    code: String,
}
#[derive(Debug, Serialize, Default, Clone)]
pub struct PythonInterpreterTool {
    pub tool: BaseTool,
}

impl PythonInterpreterTool {
    pub fn new() -> Self {
        PythonInterpreterTool {
            tool: BaseTool {
                name: "python_interpreter",
                description:  "This is a tool that evaluates python code. It can be used to perform calculations."
            }}
    }           
}

impl Tool for PythonInterpreterTool {
    type Params = PythonInterpreterToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }
    fn forward(&self, arguments: PythonInterpreterToolParams) -> Result<String> {

        Ok(format!("Evaluation Result: {}", evaluate_python_code(&arguments.code, vec![], &mut HashMap::new())?))
    }
}

pub trait AnyTool: Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn forward_json(&self, json_args: serde_json::Value) -> Result<String>;
    fn tool_info(&self) -> ToolInfo;
    fn clone_box(&self) -> Box<dyn AnyTool>;
}

impl<T: Tool + Clone + 'static> AnyTool for T {
    fn name(&self) -> &'static str {
        Tool::name(self)
    }

    fn description(&self) -> &'static str {
        Tool::description(self)
    }

    fn forward_json(&self, json_args: serde_json::Value) -> Result<String> {
        let params = serde_json::from_value::<T::Params>(json_args)?;
        Tool::forward(self, params)
    }

    fn tool_info(&self) -> ToolInfo {
        ToolInfo::new::<T::Params, T>(self)
    }

    fn clone_box(&self) -> Box<dyn AnyTool> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visit_website_tool() {
        let tool = VisitWebsiteTool::new();
        let url = "www.rust-lang.org/";
        let _result = tool.forward(&url);
    }

    #[test]
    fn test_final_answer_tool() {
        let tool = FinalAnswerTool::new();
        let arguments = FinalAnswerToolParams {
            answer: "The answer is 42".to_string(),
        };
        let result = tool.forward(arguments).unwrap();
        assert_eq!(result, "The answer is 42");
    }

    #[test]
    fn test_google_search_tool() {
        let tool = GoogleSearchTool::new(None);
        let query = "What is the capital of France?";
        let result = tool.forward(query, None);
        assert!(result.contains("Paris"));
    }

    #[test]
    fn test_duckduckgo_search_tool() {
        let tool = DuckDuckGoSearchTool::new();
        let query = "What is the capital of France?";
        let result = tool.forward(query).unwrap();
        assert!(result.iter().any(|r| r.snippet.contains("Paris")));
    }
}
