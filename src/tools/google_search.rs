//! This module contains the Google search tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::base::BaseTool;
use super::tool_traits::Tool;
use anyhow::{anyhow, Context, Result};

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
        let api_key = api_key
            .or_else(|| std::env::var("SERPAPI_API_KEY").ok())
            .unwrap_or_default();

        GoogleSearchTool {
            tool: BaseTool {
                name: "google_search",
                description: "Performs a google web search for your query then returns a string of the top search results.",
            },
            api_key,
        }
    }

    fn forward(&self, query: &str, filter_year: Option<&str>) -> Result<String> {
        if self.api_key.is_empty() {
            return Err(anyhow!("SERPAPI_API_KEY missing"));
        }
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
        let resp = client
            .get("https://serpapi.com/search.json")
            .query(&params)
            .send()
            .context("Failed to send request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(anyhow!("Failed to fetch search results: HTTP {}: {}", status, text));
        }

        let results: serde_json::Value = resp.json().context("Failed to parse JSON")?;

        let organic_results = results
            .get("organic_results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                if let Some(year) = filter_year {
                    anyhow!("'organic_results' key not found for query: '{}' with filtering on year={}. Use a less restrictive query or do not filter on year.", query, year)
                } else {
                    anyhow!("'organic_results' key not found for query: '{}'. Use a less restrictive query.", query)
                }
            })?;

        if organic_results.is_empty() {
            let suffix = if let Some(year) = filter_year {
                format!(" with filter year={}", year)
            } else {
                String::new()
            };
            return Err(anyhow!("No results found for '{}'. Try with a more general query{}.", query, suffix));
        }

        let mut web_snippets = Vec::new();
        for (idx, page) in organic_results.iter().enumerate() {
            let date_published = page.get("date").map_or(String::new(), |d| {
                format!("\nDate published: {}", d.as_str().unwrap_or(""))
            });
            let source = page.get("source").map_or(String::new(), |s| {
                format!("\nSource: {}", s.as_str().unwrap_or(""))
            });
            let snippet = page.get("snippet").map_or(String::new(), |s| {
                format!("\n{}", s.as_str().unwrap_or(""))
            });

            let redacted_version = format!(
                "{}. [{}]({}){}{}\n{}",
                idx,
                page.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                page.get("link").and_then(|v| v.as_str()).unwrap_or(""),
                date_published,
                source,
                snippet
            );
            let redacted_version = redacted_version.replace("Your browser can't play this video.", "");
            web_snippets.push(redacted_version);
        }

        Ok(format!("## Search Results\n{}", web_snippets.join("\n\n")))
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
        self.forward(&query, filter_year.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_google_search_tool() {
        let tool = GoogleSearchTool::new(None);
        let query = "What is the capital of France?";
        let result = tool.forward(query, None).unwrap();
        assert!(result.contains("Paris"));
    }
}
