use crate::dna::dnatensor;
use crate::vaestruct::DnaVae;
use crate::vaestruct::DnaVaeConfig;
use crate::vaestruct::*;
use anyhow::{Context, Result};
use burn::nn::Relu;
use burn::nn::activation::Activation;
use burn::train::metric::store::items::LossItem;
use burn::{
    config::Config,
    module::{Module, ModuleVisitor},
    nn::{Linear, LinearConfig},
    tensor::{
        Data, Int, Shape, Tensor,
        backend::Backend,
        ops::{Activation, TensorOps},
    },
    train::{
        LearnerBuilder, MetricEntry, TrainOutput, ValidOutput,
        metric::{AccuracyMetric, LossMetric},
    },
};
use rand::seq::SliceRandom;
use std::collections::HashMap;

/*
Gaurav Sablok
codeprog@icloud.com
*/

impl<B: Backend> DnaVae<B> {
    pub fn new(config: &DnaVaeConfig) -> Self {
        let flat_dim = config.vocab_size * config.seq_len;
        let encoder = DnaEncoder {
            fc1: LinearConfig::new(flat_dim, config.hidden_dim).init(&Default::default()),
            fc_mu: LinearConfig::new(config.hidden_dim, config.latent_dim)
                .init(&Default::default()),
            fc_logvar: LinearConfig::new(config.hidden_dim, config.latent_dim)
                .init(&Default::default()),
        };

        let decoder = DnaDecoder {
            fc1: LinearConfig::new(config.latent_dim, config.hidden_dim).init(&Default::default()),
            fc_out: LinearConfig::new(config.hidden_dim, flat_dim).init(&Default::default()),
        };

        Self {
            encoder,
            decoder,
            latent_dim: config.latent_dim,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 2>, // [batch, vocab*seq_len]
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let definedtensor = Relu::new().forward(self.encoder.fc1.forward(x));
        let mu = self.encoder.fc_mu.forward(definedtensor.clone());
        let logvar = self.encoder.fc_logvar.forward(definedtensor);
        let std = (logvar.clone().mul_scalar(0.5)).exp();
        let eps = Tensor::random_like(&mu, burn::tensor::Distribution::Normal(0.0, 1.0));
        let z = mu.clone() + eps * std;
        let recon_h = Relu::new().forward(self.decoder.fc1.forward(z.clone()));
        let recon_logits = self.decoder.fc_out.forward(recon_h);
        (recon_logits, mu, logvar, z)
    }
}

fn vae_loss<B: Backend>(
    recon_logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mu: Tensor<B, 2>,
    logvar: Tensor<B, 2>,
) -> burn::Tensor<B, 2> {
    /*
     * cross entropy construction
     */

    let recon_loss = recon_logits
        .log()
        .mul(target)
        .sum_dim(1)
        .mul_scalar(-1.0)
        .mean();

    let epsilon = Tensor::<B, 2>::random_like(&mu, burn::tensor::Distribution::Default);
    let std = (logvar * 0.5).exp();
    mu + epsilon + std
}

pub fn vaetrain(pathfile: &str, chopsizefile: &str) -> Result<()> {
    type Backend = burn::backend::Wgpu;
    type AD = Autodiff<Backend>;

    let valueunpack = dnatensor(pathfile, chopsizefile).unwrap();

    let config = DnaVaeConfig::new(200, valueunpack.1, 512usize, 64usize);
    let mut model: DnaVae<AD> = DnaVae::new(&config).into();

    let mut optim = burn::optim::AdamConfig::new().init();
    let mut best_loss = f64::INFINITY;
    for epoch in 0..50 {
        let x = valueunpack.0;
        let (recon, mu, logvar, _) = model.forward(x.clone());
        let loss = vae_loss(recon, x, mu, logvar);
        /*

        let grads = loss;
        let grads = optim.step(&mut model, grads);

        let loss_val = loss.into_scalar();
        println!("Epoch {:3} | loss = {:.4}", epoch, loss_val);

        if loss_val < best_loss {
            best_loss = loss_val;
            only this part to code is left and how to implement that i am finishing right now.
        }
        */
    }

    Ok(())
}
