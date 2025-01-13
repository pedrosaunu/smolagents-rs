use std::collections::HashMap;

use reqwest::blocking::Client;
use serde::Serialize;
use anyhow::Result;
use serde_json::json;
use std::fmt::Debug;

use crate::tools::{get_json_schema, Tool};

#[derive(Debug, Serialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    ToolCall,
    ToolResponse
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String
}

pub trait Model <T:Tool>:Debug {
    fn run(&self, messages: Vec<Message>, tools_to_call_from: Vec<T>, max_tokens: Option<usize>, args: Option<HashMap<String, String>>) -> Result<String>;
}

#[derive(Debug)]
pub struct OpenAIServerModel {
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
}

impl OpenAIServerModel {
    pub fn new(model_id: Option<&str>, temperature: Option<f32>, api_key: Option<String>) -> Self {
        let api_key = match api_key {
            Some(key) => key,
            None => std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"),
        };
        let model_id = model_id.unwrap_or("gpt-4o-mini");

        let client = reqwest::blocking::Client::new();
        OpenAIServerModel {
            model_id: model_id.to_string(),
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
        }
    }

}

impl<T: Tool> Model<T> for OpenAIServerModel {
    fn run(&self, messages: Vec<Message>, tools_to_call_from: Vec<T>, max_tokens: Option<usize>, args: Option<HashMap<String, String>>) -> Result<String> {
        let max_tokens = max_tokens.unwrap_or(1500);
        let messages = messages.iter().map(|message| {
            json!({
                "role": message.role,
                "content": message.content
            })
        }).collect::<Vec<_>>();

        let tools = tools_to_call_from.into_iter().map(|tool|{
            get_json_schema(tool)
        }).collect::<Vec<_>>();

        let body = json!({
            "model": self.model_id,
            "prompt": messages,
            "temperature": self.temperature,
            "tools": tools,
            "max_tokens": max_tokens,
        });
        match args {
            Some(args) => {
                let mut body = body.as_object().unwrap().clone();
                for (key, value) in args {
                    body.insert(key, json!(value));
                }
            },
            None => {}
        }
        let response = self.client.post("https://api.openai.com/v1/engines/davinci-codex/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()?
            .text()?;
        Ok(response)
    }
}