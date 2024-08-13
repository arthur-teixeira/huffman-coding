use std::{
    collections::{BinaryHeap, HashMap},
    env,
    error::Error,
    fs,
    rc::Rc,
};

#[derive(Debug)]
pub struct Prob(char, f64);

impl PartialEq for Prob {
    fn eq(&self, other: &Self) -> bool {
        self.1.eq(&other.1)
    }
}
impl Eq for Prob {}

impl PartialOrd for Prob {
    fn lt(&self, other: &Self) -> bool {
        self.1.lt(&other.1)
    }

    fn le(&self, other: &Self) -> bool {
        self.1.le(&other.1)
    }

    fn gt(&self, other: &Self) -> bool {
        self.1.gt(&other.1)
    }

    fn ge(&self, other: &Self) -> bool {
        self.1.ge(&other.1)
    }

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.1.partial_cmp(&self.1) // Reversed because it will be used in a Min-heap
    }
}

impl Ord for Prob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1.partial_cmp(&other.1).unwrap()
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HuffmanTree {
    Leaf(Prob),
    Node(Rc<HuffmanTree>, Rc<HuffmanTree>),
}

fn enqueue_chars(input: &str) -> BinaryHeap<HuffmanTree> {
    let mut char_freq: HashMap<char, u64> = HashMap::new();
    let universe = input.len();

    for ch in input.chars() {
        char_freq.entry(ch).and_modify(|v| *v += 1).or_insert(1);
    }

    char_freq
        .iter()
        .map(|(k, v)| HuffmanTree::Leaf(Prob(*k, *v as f64 / universe as f64)))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <input-file> <output-file>", args[0]);
        return Ok(());
    }

    let input_path = &args[1];
    let input = fs::read_to_string(input_path)?;

    let ordered_chars = enqueue_chars(&input);
    dbg!(ordered_chars);

    Ok(())
}

#[cfg(test)]
mod huffman_test {
    use crate::{
        enqueue_chars,
        HuffmanTree,
        Prob
    };


    #[test]
    fn test_char_freq() {
        let sut = "aaabbc";
        let mut queue = enqueue_chars(sut);
        assert_eq!(queue.pop(), Some(HuffmanTree::Leaf(Prob('c', 1f64/6f64))));
        assert_eq!(queue.pop(), Some(HuffmanTree::Leaf(Prob('b', 1f64/3f64))));
        assert_eq!(queue.pop(), Some(HuffmanTree::Leaf(Prob('a', 1f64/2f64))));
    }
}
