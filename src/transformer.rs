use anyhow::Result;
use burn::lr_scheduler::noam::{NoamLrScheduler, NoamLrSchedulerConfig};
use burn::lr_scheduler::LrScheduler;
use burn::nn::transformer::attention::multi_head::MultiHeadAttentionConfig;
use burn::optim::{lr_scheduler, AdamConfig};
use burn::tensor::{activation, Device};
use burn::{
    config::Config,
    module::Module,
    nn::{
        ::{Embedding, EmbeddingConfig},
        Linear, LinearConfig,
        transformer::{DecoderConfig, DecoderLayerConfig, pos_encoding::PositionalEncoding,
        },
    },
    tensor::{
        Data, Shape, Tensor,
        backend::{Autodiff, Backend},
        ops::{IntTensor, TensorOps},
    },
    train::{
        LearnerBuilder, TrainingInterrupter, ValidOutput,
        metric::{AccuracyMetric, LossMetric},
    },
};
use rand::seq::IndexedRandom;
use rand::RngExt;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::error::Error;
use std::io::{BufRead, BufReader};


/*
Gaurav Sablok
codeprog@icloud.com
*/

/*
Bee vocabulary building
*/

const VOCAB: [&str; 8] = ["A", "C", "G", "T", "<PAD>", "<UNK>", "<BOS>", "<EOS>"];
const PAD_TOKEN: usize = 4;
const BOS_TOKEN: usize = 6;
const EOS_TOKEN: usize = 7;

pub fn build_vocab_map() -> HashMap<String, usize> {
    VOCAB
        .iter()
        .enumerate()
        .map(|(i, &s)| (s.to_string(), i))
        .collect()
}

pub fn tokenize(seq: &str, vocab: &HashMap<String, usize>, max_len: usize) -> Vec<usize> {
    let mut tokens = vec![BOS_TOKEN];
    for c in seq.chars() {
        let token = match c.to_ascii_uppercase().to_string().as_str() {
            "A" | "C" | "G" | "T" => vocab.get(&c.to_string()).copied().unwrap_or(5),
            _ => 5, // <UNK>
        };
        if tokens.len() < max_len - 1 {
            tokens.push(token);
        }
    }
    tokens.push(EOS_TOKEN);
    while tokens.len() < max_len {
        tokens.push(PAD_TOKEN);
    }
    tokens
}

pub fn readfasta(pathfile: &str, vocab, vocab: &HashMap<String, usize>, max_len: usize) -> Result<Vec<usize>, Box<dyn Error>>{
 let fileopen = File::open(pathfile).expect("file not present");
 let fileread =BufReader::new(fileopen);
 let mut stringvec: Vec<String> = Vec::new();
 for i in fileread.lines(){
   let line = i.expect("file not present");
   if !line.starts_with(">"){
   stringvec.push(line);
   }
 }
 let valuereturn:Vec<usize> = stringvec.iter().map(|x| tokenize(x, vocab, max_len)).collect::<Vec<usize>>();
   Ok(valueturn)
 }


pub fn detokenize(tokens: &[usize]) -> String {
    tokens
        .iter()
        .filter(|&&t| t != PAD_TOKEN && t != BOS_TOKEN && t != EOS_TOKEN)
        .map(|&t| VOCAB.get(t).unwrap_or(&"<UNK>").to_string())
        .collect()
}

#[derive(Config, Debug)]
struct DnaTransformerConfig {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    max_seq_len: usize,
    dropout: f64,
}

#[derive(Module, Debug)]
struct DnaTransformer<B: Backend> {
    embedding: Embedding<B>,
    pos_encoding: PositionalEncoding<B>,
    decoder: DecoderConfig<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> DnaTransformer<B> {
    fn new(config: &DnaTransformerConfig) -> Self {
        let embedding =
            EmbeddingConfig::new(config.vocab_size, config.d_model).init(&Default::default());

        let pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len);

        let decoder = DecoderConfig::new(
            config.d_model,
            MultiHeadAttentionConfig::new(config.d_model, config.n_heads),
            DecoderLayerConfig::default().with_ffn_hidden_dim(config.d_model * 4),
            config.n_layers,
            config.dropout,
        )
        .init();

        let lm_head =
            LinearConfig::new(config.d_model, config.vocab_size).init(&Default::default());

        Self {
            embedding,
            pos_encoding,
            decoder,
            lm_head,
        }
    }

    fn forward(&self, tokens: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = tokens.dims();

        let embedded = self.embedding.forward(tokens);
        let pos = self.pos_encoding.forward(seq_len);
        let x = embedded + pos.unsqueeze_dim(0);

        // Causal mask: upper triangle masked
        let mask = (Tensor::arange(0..seq_len as i64, &B::Device::default())
            .unsqueeze_dim(0)
            .expand([seq_len as i64, seq_len as i64])
            .transpose()
            .greater_equal(
                Tensor::arange(0..seq_len as i64, &B::Device::default()).unsqueeze_dim(1),
            ))
        .into_data()
        .iter()
        .into_iter()
        .map(|v| if v { f32::NEG_INFINITY } else { 0.0 })
        .collect::<Vec<_>>();

        let causal_mask = Tensor::from_data(
            Data::new(mask, Shape::new([seq_len, seq_len])),
            burn::tensor::Device::default(),
        );
        let decoded = self.decoder.forward(x, None, Some(causal_mask)); // ignore cross-attn src
        self.lm_head.forward(decoded) // [B, S, V]
    }
}

fn loss_fn<B: Backend>(logits: Tensor<B, 3>, targets: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch, seq, vocab] = logits.dims();
    let logits_flat = logits.reshape([batch * seq, vocab]);
    let targets_flat = targets.reshape([batch * seq]);
    let loss = activation::prelu(logits_flat, targets_flat);
    let pad_mask = targets_flat.equal(Tensor::from_ints([PAD_TOKEN as usize], &Device::default()));
    let valid_count = pad_mask.into_data().iter().sum().value[0] as f32;
    if valid_count > 0.0 {
        loss / Tensor::from_floats([valid_count], &Device::default());
    } else {
        loss;
    }
    loss
}


fn generate_batch<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    device: B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut rng = rand::thread_rng();
    let alphabet = [0usize, 1, 2, 3];
    let mut inputs = vec![];
    let mut targets = vec![];
    for _ in 0..batch_size {
        let mut seq: Vec<usize> = (0..seq_len - 2)
            .map(|_| *alphabet.choose(&mut rng).unwrap())
            .collect();
        seq.insert(0, BOS_TOKEN);
        seq.push(EOS_TOKEN);

        let target = seq[1..].to_vec();

        inputs.push(seq);
        targets.push(target);
    }

    let input_data = Data::new(
        inputs.into_iter().flatten().map(|t| t as i64).collect(),
        Shape::new([batch_size, seq_len]),
    );
    let target_data = Data::new(
        targets.into_iter().flatten().map(|t| t as i64).collect(),
        Shape::new([batch_size, seq_len]),
    );

    (
        Tensor::from_data(input_data, &Device::default()).to_device(&device),
        Tensor::from_data(target_data, &Device::default()).to_device(&device),
    )
}

fn beetransformer() -> Result<(), Box<dyn Error>> {
    type B = burn::backend::Wgpu;
    type AD = Autodiff<B>;

    let device =  Device::default();
    let config = DnaTransformerConfig::new(8, 128, 4, 3, 128, 0.1);
    let mut model: DnaTransformer<AD> = DnaTransformer::new(&config).to_device(&device);
    let optim = AdamConfig::new().init();
    let mut learner = LearnerBuilder::new("bee_transform").num_epochs(100).metric_train_numeric(AccuracyMetric::new()).metric_valid_numeric(AccuracyMetric::new()).metric_train(LossMetric::new()).metric_valid(LossMetric::new()).checkpoint(100usize).build(model, optim, NoamLrSchedulerConfig::new(1e-3)));

    let batch_size = 64;
    let seq_len = 1000;
    for epoch in 1..=100 {
        let mut total_loss = 0.0;
        let steps = 200;
        for step in 0..steps {
            let (inputs, targets) = generate_batch(batch_size, seq_len, device.clone());
            let output = learner.model().forward(inputs.clone());
            let loss = loss_fn(output, targets.clone());
            let grads = loss.backward();
            learner.optim().step(&mut learner.model(), grads);
            total_loss += loss.into_scalar();
            if step % 20 == 0 {
                println!(
                    "Epoch {:2} | Step {:3}/{:3} | loss = {:.4}",
                    epoch,
                    step,
                    steps,
                    loss.into_scalar()
                );
            }
        }
        println!(
            "Epoch {:2} completed | avg train loss = {:.4}",
            epoch,
            total_loss / steps as f64
        );
        if epoch % 5 == 0 {
            let sample_input =
                Tensor::from_ints([[BOS_TOKEN as i64]], &device).expand([1, seq_len]);
            let logits = learner.model().forward(sample_input);
            let predicted = logits
                .argmax_dim(2)
                .into_data()
                .value
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>();

            let generated = detokenize(&predicted);
            println!("Sample generation: {}", generated);
        }
    }
    Ok(())
}
