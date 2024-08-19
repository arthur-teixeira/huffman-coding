use std::{
    collections::{BinaryHeap, HashMap},
    env,
    error::Error,
    fs,
    rc::Rc,
};

#[derive(Debug, Clone, Copy)]
pub struct Prob(f64);

// /\<1\>/0/gI
impl PartialEq for Prob {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl Eq for Prob {}

impl PartialOrd for Prob {
    fn lt(&self, other: &Self) -> bool {
        self.0.lt(&other.0)
    }

    fn le(&self, other: &Self) -> bool {
        self.0.le(&other.0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.0.gt(&other.0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.0.ge(&other.0)
    }

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.0.partial_cmp(&self.0) // Reversed because it will be used in a Min-heap
    }
}

impl Ord for Prob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HuffmanTree {
    Leaf(char, Prob),
    Node(Prob, Rc<HuffmanTree>, Rc<HuffmanTree>),
}

impl HuffmanTree {
    fn prob(&self) -> f64 {
        match self {
            Self::Leaf(_, p) => p.0,
            Self::Node(p, ..) => p.0,
        }
    }
}

fn enqueue_chars(input: &str) -> BinaryHeap<HuffmanTree> {
    let mut char_freq: HashMap<char, u64> = HashMap::new();
    let universe = input.len();

    for ch in input.chars() {
        char_freq.entry(ch).and_modify(|v| *v += 1).or_insert(1);
    }

    char_freq
        .iter()
        .map(|(k, v)| HuffmanTree::Leaf(*k, Prob(*v as f64 / universe as f64)))
        .collect()
}

fn construct_huffman_tree(mut freqs: BinaryHeap<HuffmanTree>) -> HuffmanTree {
    while freqs.len() > 1 {
        let a = freqs.pop().unwrap();
        let b = freqs.pop().unwrap();
        let new = HuffmanTree::Node(Prob(a.prob() + b.prob()), a.into(), b.into());
        freqs.push(new);
    }

    freqs
        .pop()
        .expect("Expected one node to remain in the queue")
}

fn print_tree(t: &HuffmanTree) {
    match t {
        HuffmanTree::Leaf(c, Prob(p)) => print!("{}: {} ", c, p),
        HuffmanTree::Node(_, a, b) => {
            print_tree(a);
            println!();
            print_tree(b);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <input-file> <output-file>", args[0]);
        return Ok(());
    }

    let input_path = &args[1];
    let input = fs::read_to_string(input_path)?;
    let input = "A_DEAD_DAD_CEDED_A_BAD_BABE_A_BEADED_ABACA_BED";

    let queue = enqueue_chars(&input);
    let tree = construct_huffman_tree(queue);
    print_tree(&tree);

    Ok(())
}

#[cfg(test)]
mod huffman_test {
    use crate::{enqueue_chars, HuffmanTree, Prob};

    #[test]
    fn test_char_freq() {
        let sut = "aaabbc";
        let mut queue = enqueue_chars(sut);
        assert_eq!(queue.pop(), Some(HuffmanTree::Leaf('c', Prob(1f64 / 6f64))));
        assert_eq!(queue.pop(), Some(HuffmanTree::Leaf('b', Prob(1f64 / 3f64))));
        assert_eq!(queue.pop(), Some(HuffmanTree::Leaf('a', Prob(1f64 / 2f64))));
    }
}
