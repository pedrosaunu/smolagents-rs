use anyhow::Result;
use htmd::HtmlToMarkdown;
use reqwest::Url;
use schemars::schema::RootSchema;
use scraper::Selector;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use schemars::JsonSchema;
pub trait Parameters: DeserializeOwned + JsonSchema {}
pub trait Tool: Debug {
    type Params: Parameters;
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>>;
    fn output_type(&self) -> &'static str;
    fn is_initialized(&self) -> bool;
    fn forward(&self, arguments: Self::Params) -> Result<Box<dyn Any>>;
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

#[derive( Serialize)]
pub struct ToolInfo {
    #[serde(rename = "type")]
    tool_type: ToolType,
    pub function: ToolFunctionInfo,
}

#[derive(Serialize)]
pub struct ToolFunctionInfo {
    pub name: &'static str,
    description: &'static str,
    parameters: RootSchema,
}


pub fn get_json_schema(tool: &ToolInfo) -> serde_json::Value {
    json!(tool)
}

pub trait ToolGroup {
    fn tools(&self) -> Vec<Box<dyn Tool>>;
}

#[derive(Deserialize, JsonSchema)]
struct BaseParams{
    #[schemars(description = "The name of the tool")]
    name: String,
}
impl<P: DeserializeOwned + JsonSchema> Parameters for P {}


#[derive(Debug, Serialize)]
pub struct BaseTool {
    pub name: &'static str,
    pub description: &'static str,
    pub inputs: HashMap<&'static str, HashMap<&'static str, String>>,
    pub output_type: &'static str,
    pub is_initialized: bool,
}

impl Tool for BaseTool {
    type Params = BaseParams;
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> {
        &self.inputs
    }

    fn output_type(&self) -> &'static str {
        self.output_type
    }

    fn is_initialized(&self) -> bool {
        self.is_initialized
    }
    fn forward(&self, _arguments: BaseParams) -> Result<Box<dyn Any>> {
        Ok(Box::new("Not implemented"))
    }
}
#[derive(Debug, Serialize)]
pub struct VisitWebsiteTool {
    pub tool: BaseTool,
}
impl Default for VisitWebsiteTool {
    fn default() -> Self {
        VisitWebsiteTool::new()
    }
}

impl VisitWebsiteTool {
    pub fn new() -> Self {
        VisitWebsiteTool {
            tool: BaseTool {
                name: "visit_website",
                description: "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages",
                inputs: HashMap::from([
                    ("url", HashMap::from([
                        ("type", "string".to_string()),
                        ("description", "url of the webpage to visit".to_string()),
                        ("required", "true".to_string()),
                    ])),
                ]),
                output_type: "string",
                is_initialized: false,
            },
        }
    }

    pub fn forward(&self, url: &str) -> String {
        let client = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());

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
struct VisitWebsiteToolParams{
    #[schemars(description = "The url of the website to visit")]
    url: String,
}

impl Tool for VisitWebsiteTool {
    type Params = VisitWebsiteToolParams;
    fn name(&self) -> &'static str {
        self.tool.name()
    }

    fn description(&self) -> &'static str {
        self.tool.description()
    }

    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> {
        self.tool.inputs()
    }

    fn output_type(&self) -> &'static str {
        self.tool.output_type()
    }

    fn is_initialized(&self) -> bool {
        self.tool.is_initialized()
    }

    fn forward(&self, arguments: VisitWebsiteToolParams) -> Result<Box<dyn Any>> {
        let url = arguments.url;
        Ok(Box::new(self.forward(&url)))
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct FinalAnswerToolParams{
    #[schemars(description = "The final answer to the problem")]
    answer: String,
}

#[derive(Debug, Serialize)]
pub struct FinalAnswerTool {
    pub tool: BaseTool,
}
impl Default for FinalAnswerTool {
    fn default() -> Self {
        FinalAnswerTool::new()
    }
}
impl FinalAnswerTool {
    pub fn new() -> Self {
        FinalAnswerTool {
            tool: BaseTool {
                name: "final_answer",
                description: "Provides a final answer to the given problem.",
                inputs: HashMap::from([(
                    "answer",
                    HashMap::from([
                        ("type", "string".to_string()),
                        ("description", "The final answer to the problem".to_string()),
                        ("required", "true".to_string()),
                    ]),
                )]),
                output_type: "string",
                is_initialized: false,
            },
        }
    }
}

impl Tool for FinalAnswerTool {
    type Params = FinalAnswerToolParams;
    fn name(&self) -> &'static str {
        self.tool.name()
    }
    fn description(&self) -> &'static str {
        self.tool.description()
    }
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> {
        self.tool.inputs()
    }
    fn output_type(&self) -> &'static str {
        self.tool.output_type()
    }
    fn is_initialized(&self) -> bool {
        self.tool.is_initialized()
    }

    fn forward(&self, arguments: FinalAnswerToolParams) -> Result<Box<dyn Any>> {
        let answer = arguments.answer;
        Ok(Box::new(answer.to_string()))
    }
}

#[derive(Deserialize, JsonSchema)]
struct GoogleSearchToolParams{
    #[schemars(description = "The query to search for")]
    query: String,
    #[schemars(description = "Optionally restrict results to a certain year")]
    filter_year: Option<String>,
}

#[derive(Debug)]
pub struct GoogleSearchTool {
    pub tool: BaseTool,
    pub api_key: String,
}

impl Default for GoogleSearchTool {
    fn default() -> Self {
        GoogleSearchTool::new(None)
    }
}

impl GoogleSearchTool {
    pub fn new(api_key: Option<String>) -> Self {
        let api_key = api_key.unwrap_or(std::env::var("SERPAPI_API_KEY").unwrap());

        GoogleSearchTool {
            tool: BaseTool {
                name: "google_search",
                description: "Performs a google web search for your query then returns a string of the top search results.",
                inputs: HashMap::from([
                    ("query", HashMap::from([
                        ("type", "string".to_string()),
                        ("description", "The query to search for".to_string()),
                        ("required", "true".to_string()),
                    ])),
                    ("filter_year", HashMap::from([
                        ("type", "string".to_string()),
                        ("description", "Optionally restrict results to a certain year".to_string()),
                        ("required", "false".to_string())
                    ])),
                ]),
                output_type: "string",
                is_initialized: false,
            },
            api_key,
        }
    }

    fn forward(&self, query: &str, filter_year: Option<&str>) -> String {
        let params = {
            let mut params = HashMap::new();
            params.insert("engine", "google".to_string());
            params.insert("q", query.to_string());
            params.insert("api_key", self.api_key.clone());
            params.insert("google_domain", "google.com".to_string());

            if let Some(year) = filter_year {
                params.insert(
                    "tbs",
                    format!("cdr:1,cd_min:01/01/{},cd_max:12/31/{}", year, year),
                );
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
        self.tool.name()
    }
    fn description(&self) -> &'static str {
        self.tool.description()
    }
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> {
        self.tool.inputs()
    }
    fn output_type(&self) -> &'static str {
        self.tool.output_type()
    }
    fn is_initialized(&self) -> bool {
        self.tool.is_initialized()
    }

    fn forward(&self, arguments: GoogleSearchToolParams) -> Result<Box<dyn Any>> {
        let query = arguments.query;
        let filter_year = arguments.filter_year;
        Ok(Box::new(self.forward(&query, filter_year.as_deref())))
    }
}

#[derive(Deserialize, JsonSchema)]
struct DuckDuckGoSearchToolParams{
    #[schemars(description = "The query to search for")]
    query: String,
}



#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct DuckDuckGoSearchTool {
    pub tool: BaseTool,
}

impl DuckDuckGoSearchTool {
    pub fn new() -> Self {
        DuckDuckGoSearchTool {
            tool: BaseTool {
                name: "duckduckgo_search",
                description: "Performs a duckduckgo web search for your query then returns a string of the top search results.",
                inputs: HashMap::from([
                    ("query", HashMap::from([
                        ("type", "string".to_string()),
                        ("description", "The query to search for".to_string()),
                        ("required", "true".to_string()),
                    ])),
                ]),
                output_type: "string",
                is_initialized: false,
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
        self.tool.name()
    }
    fn description(&self) -> &'static str {
        self.tool.description()
    }
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> {
        self.tool.inputs()
    }
    fn output_type(&self) -> &'static str {
        self.tool.output_type()
    }
    fn is_initialized(&self) -> bool {
        self.tool.is_initialized()
    }
    fn forward(&self, arguments: DuckDuckGoSearchToolParams) -> Result<Box<dyn Any>> {
        let query = arguments.query;
        let results = self.forward(&query)?;
        let json_string = serde_json::to_string_pretty(&results)?;
        Ok(Box::new(json_string))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visit_website_tool() {
        let tool = VisitWebsiteTool::new();
        let url = "https://www.rust-lang.org/";
        let _result = tool.forward(&url);
    }

    #[test]
    fn test_final_answer_tool() {
        let tool = FinalAnswerTool::new();
        let arguments = FinalAnswerToolParams{answer: "The answer is 42".to_string()};
        let result = tool.forward(arguments).unwrap();
        assert_eq!(result.downcast_ref::<String>().unwrap(), "The answer is 42");
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
