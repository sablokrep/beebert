mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod dna;
mod vaestruct;
mod var;
use crate::var::vaetrain;

/*
Gaurav Sablok,
codeprog@icloud.com
*/

fn main() {
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::VAEEncoder {
            fastafile,
            lengthkeep,
            threadnt,
        } => {
            let nthreads = threadnt.parse::<usize>().unwrap();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads)
                .build()
                .unwrap();
            pool.install(|| {
                let command = vaetrain(fastafile, lengthkeep);
                println!("The command has finished:{}", command);
            });
        }
    }
}
