//! This module contains the tools that can be used in an agent. These are the default tools that are available.
//! You can also implement your own tools by implementing the `Tool` trait.

pub mod tool_traits;
pub mod base;
pub mod visit_website;
pub mod final_answer;
pub mod google_search;
pub mod ddg_search;

#[cfg(feature = "code-agent")]
pub mod python_interpreter;

pub use tool_traits::*;
pub use base::*;
pub use visit_website::*;
pub use final_answer::*;
pub use google_search::*;
pub use ddg_search::*;

#[cfg(feature = "code-agent")]
pub use python_interpreter::*;