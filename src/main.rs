use std::{
    collections::{BinaryHeap, HashMap},
    env,
    error::Error,
    fs,
    rc::Rc,
};

const BYTE_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialOrd)]
pub struct Prob(f64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CharCode(usize, u8);

impl PartialEq for Prob {
    fn eq(&self, other: &Self) -> bool {
        other.0.eq(&self.0)
    }
}
impl Eq for Prob {}

impl Ord for Prob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum HuffmanTree {
    Leaf(char, Prob),
    Node(Prob, Rc<HuffmanTree>, Rc<HuffmanTree>),
}

impl PartialOrd for HuffmanTree {
    fn lt(&self, other: &Self) -> bool {
        other.prob().lt(&self.prob())
    }

    fn le(&self, other: &Self) -> bool {
        other.prob().le(&self.prob())
    }

    fn gt(&self, other: &Self) -> bool {
        other.prob().gt(&self.prob())
    }

    fn ge(&self, other: &Self) -> bool {
        other.prob().ge(&self.prob())
    }

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.prob().partial_cmp(&self.prob())
    }
}

impl Ord for HuffmanTree {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl HuffmanTree {
    fn prob(&self) -> f64 {
        match self {
            Self::Leaf(_, p) => p.0,
            Self::Node(p, ..) => p.0,
        }
    }

    fn construct_dictionary(&self, stack: &mut Vec<u8>, result: &mut HashMap<char, CharCode>) {
        match self {
            HuffmanTree::Leaf(c, _) => {
                let mut snapshot = stack.clone();
                let mut code: usize = 0;
                while snapshot.len() > 0 {
                    let i = snapshot.pop().unwrap();
                    code = (code << 1) | i as usize;
                }
                result.insert(*c, CharCode(code, stack.len() as u8));
            }
            HuffmanTree::Node(_, l, r) => {
                stack.push(0);
                l.construct_dictionary(stack, result);
                stack.pop();

                stack.push(1);
                r.construct_dictionary(stack, result);
                stack.pop();
            }
        }
    }
}

struct Bitfield {
    bf: Vec<u8>,
    len: usize,
    pos: usize,
}

impl Bitfield {
    fn new() -> Self {
        Self {
            bf: Vec::new(),
            pos: 0,
            len: 0,
        }
    }

    fn set_bit(&mut self, val: u8) {
        if self.bf.len() <= (self.pos / BYTE_SIZE) {
            self.bf.push(0);
        }
        self.bf[self.pos / BYTE_SIZE] |= val << (BYTE_SIZE - ((self.pos) % BYTE_SIZE) - 1);
        self.pos += 1;
    }

    fn write(&mut self, CharCode(code, mut code_len): &CharCode) {
        while code_len > 0 {
            self.set_bit((code >> (code_len - 1)) as u8 & 1);
            code_len -= 1;
        }
    }

    fn next_bit(&mut self) -> Option<u8> {
        if self.pos / BYTE_SIZE + (BYTE_SIZE - (self.pos % BYTE_SIZE) - 1) > self.len {
            return None;
        }

        let a = !!(self.bf)[self.pos / BYTE_SIZE] & (1 << (BYTE_SIZE - ((self.pos) % BYTE_SIZE) - 1));
        self.pos += 1;

        Some(a)
    }
}

impl Iterator for Bitfield {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_bit()
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

fn compress(original: &str, dict: &HashMap<char, CharCode>) -> Bitfield {
    let mut bf = Bitfield::new();

    for c in original.chars() {
        let code = dict.get(&c).expect("Character is not in dictionary");
        bf.write(code);
    }

    bf
}

fn decompress(compressed: &mut Bitfield, dict: &HashMap<char, CharCode>) -> String {
    compressed.len = compressed.pos;
    compressed.pos = 0;
    for bit in compressed {
        println!("{:#01b}", bit);
    }

    "".into()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <input-file> <output-file>", args[0]);
        return Ok(());
    }

    let input_path = &args[1];
    let input = fs::read_to_string(input_path)?;

    let queue = enqueue_chars(&input);
    let tree = construct_huffman_tree(queue.into());

    let mut s = Vec::new();
    let mut dict = HashMap::new();
    tree.construct_dictionary(&mut s, &mut dict);
    let mut compressed = compress(&input, &dict);

    decompress(&mut compressed, &dict);

    Ok(())
}

#[cfg(test)]
mod huffman_test {
    use crate::{compress, construct_huffman_tree, enqueue_chars, HuffmanTree, Prob};
    use std::collections::HashMap;

    fn get_compressed(sut: &str) -> crate::Bitfield {
        let queue = enqueue_chars(sut);
        let tree = construct_huffman_tree(queue);
        let mut dict = HashMap::new();
        let mut stack = Vec::new();
        tree.construct_dictionary(&mut stack, &mut dict);
        for c in sut.chars() {
            let code = dict.get(&c).unwrap();
            println!("char {} has code {:#08b} with len {}", c, code.0, code.1);
        }

        compress(sut, dict)
    }

    #[test]
    fn test_char_freq() {
        let sut = "ddddeeeeeaaabbc";
        let mut queue = enqueue_chars(sut);
        assert_eq!(
            queue.pop(),
            Some(HuffmanTree::Leaf('c', Prob(1f64 / sut.len() as f64)))
        );
        assert_eq!(
            queue.pop(),
            Some(HuffmanTree::Leaf('b', Prob(2f64 / sut.len() as f64)))
        );
        assert_eq!(
            queue.pop(),
            Some(HuffmanTree::Leaf('a', Prob(3f64 / sut.len() as f64)))
        );
    }

    #[test]
    fn test_reverse_tree_cmp() {
        let a = HuffmanTree::Leaf('a', Prob(1f64));
        let b = HuffmanTree::Leaf('b', Prob(3f64));

        assert_eq!(a < b, !(1 < 3));
    }

    #[test]
    fn test_compression() {
        let compressed = get_compressed("AAB");
        assert_eq!(compressed.bf, vec![0b11000000]);
        assert_eq!(compressed.pos, 3);

        let compressed = get_compressed("AAABBA");
        assert_eq!(compressed.bf, vec![0b11100100]);
        assert_eq!(compressed.pos, 6);

        let compressed = get_compressed("AAABB");
        assert_eq!(compressed.bf, vec![0b11100000]);
        assert_eq!(compressed.pos, 5);

        let compressed = get_compressed("AAAAAAABA");
        assert_eq!(compressed.bf, vec![0b11111110, 0b10000000]);
        assert_eq!(compressed.pos, 9);

        let compressed = get_compressed("AAAACCCB");
        assert_eq!(compressed.bf, vec![0b00001111, 0b11010000]);
        assert_eq!(compressed.pos, 12);
    }
}
