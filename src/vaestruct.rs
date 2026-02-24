use anyhow::{Context, Result};
use burn::{config::Config, module::Module, nn::Linear, tensor::backend::Backend};

#[derive(Config, Debug)]
pub struct DnaVaeConfig {
    pub seq_len: usize,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
}

impl DnaVaeConfig {
    pub fn newconfig(seq_len: usize) -> Self {
        Self {
            seq_len,
            vocab_size: 4,
            hidden_dim: 512,
            latent_dim: 32,
        }
    }
}

#[derive(Module, Debug)]
pub struct DnaEncoder<B: Backend> {
    pub fc1: Linear<B>,
    pub fc_mu: Linear<B>,
    pub fc_logvar: Linear<B>,
}

#[derive(Module, Debug)]
pub struct DnaDecoder<B: Backend> {
    pub fc1: Linear<B>,
    pub fc_out: Linear<B>,
}

#[derive(Module, Debug)]
pub struct DnaVae<B: Backend> {
    pub encoder: DnaEncoder<B>,
    pub decoder: DnaDecoder<B>,
    pub latent_dim: usize,
}
