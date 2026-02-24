use burn::backend::Wgpu;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
/*
Gaurav Sablok
codeprog@icloud.com
*/

/*
a chopped 2d tensor for Wgpu
*/

type Backendtype = Wgpu;

pub fn dnatensor(
    pathfile: &str,
    chopsize: &str,
) -> Result<(Tensor<Backendtype, 2>, usize), Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut filestring: Vec<String> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("line not present");
        if !line.starts_with(">") {
            let linechop = line[0..chopsize.parse::<usize>().unwrap()].to_string();
            filestring.push(linechop);
        }
    }

    let mut valuespec: Vec<Vec<f64>> = Vec::new();
    for i in filestring.iter() {
        let valuechar = i.to_uppercase().chars().collect::<Vec<_>>();
        let mut matchvec: Vec<f64> = Vec::new();
        for i in valuechar.iter() {
            match i {
                'A' => matchvec.push(0.0),
                'T' => matchvec.push(1.0),
                'C' => matchvec.push(2.0),
                'G' => matchvec.push(3.0),
                _ => continue,
            }
        }
        valuespec.push(matchvec);
    }

    let batchsize = valuespec.len();
    let dimension = valuespec[0].len();
    let untanglevec = valuespec.iter().cloned().flatten().collect::<Vec<f64>>();

    let initialtensormake = TensorData::new(untanglevec.to_vec(), [batchsize, dimension]);
    let initialtensor = Tensor::<Backendtype, 2>::from_data(initialtensormake, &Default::default());
    Ok((initialtensor, dimension))
}
