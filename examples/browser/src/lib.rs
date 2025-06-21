use wasm_bindgen::prelude::*;
use smolagents_rs::tools::tree_sitter_tool::{TreeSitterTool, TreeSitterToolParams};

#[wasm_bindgen]
pub fn parse_rust_to_ast(code: &str) -> String {
    let tool = TreeSitterTool::new();
    let params = TreeSitterToolParams { code: code.to_string() };
    tool.forward(params).unwrap_or_else(|e| format!("error: {}", e))
}
