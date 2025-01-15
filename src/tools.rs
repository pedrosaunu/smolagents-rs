use anyhow::Result;
use htmd::HtmlToMarkdown;
use serde::Serialize;
use serde_json::json;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

pub trait Tool: Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>>;
    fn output_type(&self) -> &'static str;
    fn is_initialized(&self) -> bool;
    fn forward(&self, arguments: HashMap<String, String>) -> Result<Box<dyn Any>>;
}

#[derive(Debug, Serialize)]
pub struct BaseTool {
    pub name: &'static str,
    pub description: &'static str,
    pub inputs: HashMap<&'static str, HashMap<&'static str, String>>,
    pub output_type: &'static str,
    pub is_initialized: bool,
}

impl Tool for BaseTool {
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
    fn forward(&self, _arguments: HashMap<String, String>) -> Result<Box<dyn Any>> {
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
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(url)
            .header("User-Agent", "Mozilla/5.0 (compatible; MyRustTool/1.0)")
            .send();

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
                } else {
                    format!("Failed to fetch the webpage: HTTP {}", resp.status())
                }
            }
            Err(e) => format!("Failed to make the request: {}", e),
        }
    }
}

impl Tool for VisitWebsiteTool {
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

    fn forward(&self, arguments: HashMap<String, String>) -> Result<Box<dyn Any>> {
        let url = arguments.get("url").unwrap();
        Ok(Box::new(self.forward(url)))
    }
}

pub fn get_json_schema(tool: &dyn Tool) -> serde_json::Value {
    let mut properties = HashMap::new();
    for (key, value) in tool.inputs().iter() {
        // Create a new HashMap without the 'required' field
        let mut clean_value = HashMap::new();
        clean_value.insert("type", value.get("type").unwrap().clone());
        clean_value.insert("description", value.get("description").unwrap().clone());
        properties.insert(*key, clean_value);
    }

    let required: Vec<String> = tool
        .inputs()
        .iter()
        .filter(|(_, value)| value.get("required").unwrap() == "true")
        .map(|(key, _)| (*key).to_string())
        .collect();

    json!({
        "type": "function",
        "function": {
            "name": tool.name(),
            "description": tool.description(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            },
        }
    })
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

    fn forward(&self, arguments: HashMap<String, String>) -> Result<Box<dyn Any>> {
        let answer = arguments.get("answer").unwrap();
        Ok(Box::new(answer.to_string()))
    }
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
                    format!("Failed to fetch search results: HTTP {}, Error: {}", resp.status(), resp.text().unwrap())
                }
            }
            Err(e) => format!("Failed to make the request: {}", e),
        }
    }
}

impl Tool for GoogleSearchTool {
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

    fn forward(&self, arguments: HashMap<String, String>) -> Result<Box<dyn Any>> {
        let query = arguments.get("query").unwrap();
        let filter_year = match arguments.contains_key("filter_year") {
            true => Some(arguments.get("filter_year").unwrap().as_str()),
            false => None,
        };
        Ok(Box::new(self.forward(query, filter_year)))
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
}
