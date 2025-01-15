use std::collections::HashMap;
use std::any::Any;
use reqwest::blocking::get;
use anyhow::Result;
use serde::Serialize;
use serde_json::json;
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
#[derive(Debug,Serialize)]
pub struct VisitWebsiteTool {
    pub tool: BaseTool,
}

impl VisitWebsiteTool {
    pub fn new() -> Box<dyn Tool> {
        Box::new(VisitWebsiteTool {
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
        })
    }

    pub fn forward(&self, url: &str) -> String {

        let response = get(url);
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text() {
                        Ok(text) => html2md::parse_html(&text),
                        Err(_) => "Failed to read response text".to_string(),
                    }
                } else {
                    format!("Failed to fetch the webpage: HTTP {}", resp.status())
                }
            }
            Err(_) => "Failed to make the request".to_string(),
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

pub fn get_json_schema(tool: &Box<dyn Tool>) -> serde_json::Value {
    let mut properties = HashMap::new();
    for (key, value) in tool.inputs().iter() {
        // Create a new HashMap without the 'required' field
        let mut clean_value = HashMap::new();
        clean_value.insert("type", value.get("type").unwrap().clone());
        clean_value.insert("description", value.get("description").unwrap().clone());
        properties.insert(*key, clean_value);
    }

    let required: Vec<String> = tool.inputs().iter()
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


impl FinalAnswerTool {
    pub fn new() -> Box<dyn Tool> {
        Box::new(FinalAnswerTool {
            tool: BaseTool {
                name: "final_answer",
                description: "Provides a final answer to the given problem.",
                inputs: HashMap::from([
                    ("answer", HashMap::from([
                        ("type", "string".to_string()),
                        ("description", "The final answer to the problem".to_string()),
                        ("required", "true".to_string()),
                    ])),
                ]), 
                output_type: "string",
                is_initialized: false,
            },
        })
    }
}


impl Tool for FinalAnswerTool {
    fn name(&self) -> &'static str { self.tool.name() }
    fn description(&self) -> &'static str { self.tool.description() }
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> { self.tool.inputs() }
    fn output_type(&self) -> &'static str { self.tool.output_type() }
    fn is_initialized(&self) -> bool { self.tool.is_initialized() }

    fn forward(&self, arguments: HashMap<String, String>) -> Result<Box<dyn Any>> {
        let answer = arguments.get("answer").unwrap();
        Ok(Box::new(answer.to_string()))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visit_website_tool() {
        let tool = VisitWebsiteTool::new();
        let url = "https://www.rust-lang.org/";
        let _result = tool.forward(HashMap::from([("url".to_string(), url.to_string())]));

    }
}