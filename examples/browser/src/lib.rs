use wasm_bindgen::prelude::*;
use smolagents_rs::tools::tree_sitter_tool::{
    CodeLanguage, TreeSitterTool, TreeSitterToolParams,
};

#[wasm_bindgen]
pub fn parse_code_to_ast(language: &str, code: &str) -> String {
    let tool = TreeSitterTool::new();
    let lang = match language {
        "python" => CodeLanguage::Python,
        "javascript" => CodeLanguage::Javascript,
        "bash" => CodeLanguage::Bash,
        _ => CodeLanguage::Rust,
    };
    let params = TreeSitterToolParams {
        code: code.to_string(),
        language: lang,
    };
    tool.forward(params).unwrap_or_else(|e| format!("error: {}", e))
}
