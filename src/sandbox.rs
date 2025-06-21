use std::path::Path;
use tempfile::{tempdir, Builder, TempDir};

/// Sandbox provides an isolated temporary directory for agent execution.
pub struct Sandbox {
    dir: TempDir,
}

impl Sandbox {
    /// Create a new sandbox directory. If the `SANDBOX_DIR` environment variable
    /// is set, the sandbox will be created inside that directory.
    pub fn new() -> std::io::Result<Self> {
        if let Ok(path) = std::env::var("SANDBOX_DIR") {
            let dir = Builder::new().prefix("smolagents-").tempdir_in(path)?;
            Ok(Self { dir })
        } else {
            let dir = tempdir()?;
            Ok(Self { dir })
        }
    }

    /// Path to the sandbox directory.
    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Set the sandbox directory as the current working directory.
    pub fn set_as_cwd(&self) -> std::io::Result<()> {
        std::env::set_current_dir(self.path())
    }
}
