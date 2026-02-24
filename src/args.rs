use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "beebert",
    version = "1.0",
    about = "     beebert: Variantional Autoencoder for
          Bee Olfactory sequence prediction
       ************************************************
       Author Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run the variantional autoencoder
    VAEEncoder {
        /// provide the file for the fasta
        fastafile: String,
        /// provide the exact length to keep
        lengthkeep: String,
        /// number of thread for minimap
        threadnt: String,
    },
}
