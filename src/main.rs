use std::{
    collections::{BinaryHeap, HashMap, VecDeque},
    env,
    error::Error,
    fs::{self, OpenOptions},
    os::unix::fs::FileExt,
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
    const NODE: u8 = 0xFF;

    fn prob(&self) -> f64 {
        match self {
            Self::Leaf(_, p) => p.0,
            Self::Node(p, ..) => p.0,
        }
    }

    fn construct_dictionary(&self, stack: &mut Vec<u8>, result: &mut HashMap<char, CharCode>) {
        match self {
            HuffmanTree::Leaf(c, _) => {
                let mut code: usize = 0;
                if stack.len() == 0 {
                    result.insert(*c, CharCode(code, 1));
                    return;
                }

                let mut snapshot = stack.clone();
                snapshot.reverse();
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

    fn serialize(&self) -> Vec<u8> {
        let mut buf: VecDeque<u8> = VecDeque::new();
        let mut queue: VecDeque<&HuffmanTree> = VecDeque::new();
        queue.push_back(self);

        while queue.len() > 0 {
            let node = queue.pop_front().unwrap();
            match node {
                HuffmanTree::Node(_, l, r) => {
                    buf.push_back(Self::NODE);
                    queue.push_back(l);
                    queue.push_back(r);
                }
                HuffmanTree::Leaf(c, _) => {
                    buf.push_back(*c as u8);
                }
            }
        }

        buf.push_front(buf.len() as u8);

        buf.into()
    }

    fn deserialize_tree<'a>(buf: &mut impl Iterator<Item = &'a u8>) -> Option<Rc<HuffmanTree>> {
        if let Some(byte) = buf.next() {
            match *byte {
                Self::NODE => {
                    let left = Self::deserialize_tree(buf)?;
                    let right = Self::deserialize_tree(buf)?;
                    Some(Rc::new(HuffmanTree::Node(Prob(0.0), left, right)))
                }
                value => Some(Rc::new(HuffmanTree::Leaf(value as char, Prob(0.0)))),
            }
        } else {
            None
        }
    }

    fn from_buffer(buf: Vec<u8>) -> Option<Rc<Self>> {
        let len = buf.get(0)?;
        if buf.len() != (len + 1) as usize {
            return None;
        }

        let tree_buf = &buf[1..=*len as usize];
        Self::deserialize_tree(&mut tree_buf.into_iter())
    }
}

#[derive(Debug, Clone)]
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
        if self.pos >= self.len {
            return None;
        }

        let a = self.bf[self.pos / BYTE_SIZE] & (1 << (BYTE_SIZE - 1 - (self.pos % BYTE_SIZE))) > 0;
        self.pos += 1;

        Some(a as u8)
    }

    fn at(&self, pos: usize) -> Option<u8> {
        if pos >= self.len {
            return None;
        }

        let a = self.bf[pos / BYTE_SIZE] & (1 << (BYTE_SIZE - 1 - (pos % BYTE_SIZE))) > 0;

        Some(a as u8)
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

fn decompress(compressed: &mut Bitfield, tree: &HuffmanTree) -> String {
    compressed.len = compressed.pos;
    compressed.pos = 0;
    let mut result = String::new();
    decompress_recur(compressed, tree, tree, 0, &mut result);

    result
}

fn decompress_recur(
    compressed: &mut Bitfield,
    root: &HuffmanTree,
    node: &HuffmanTree,
    pos: usize,
    result: &mut String,
) {
    match node {
        HuffmanTree::Leaf(c, _) => {
            if root == node {
                let a: String = std::iter::repeat(c).take(compressed.len).collect();
                result.push_str(&a);
            } else {
                result.push(*c);
                decompress_recur(compressed, root, root, pos, result);
            }
        }
        HuffmanTree::Node(_, l, r) => {
            let bit = compressed.at(pos);
            match bit {
                Some(0) => decompress_recur(compressed, root, l, pos + 1, result),
                Some(1) => decompress_recur(compressed, root, r, pos + 1, result),
                None => {}
                _ => unreachable!(),
            }
        }
    }
}

fn do_compression(input_path: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
    let input = fs::read_to_string(input_path)?;

    let queue = enqueue_chars(&input);
    let tree = construct_huffman_tree(queue.into());

    let mut s = Vec::new();
    let mut dict = HashMap::new();
    tree.construct_dictionary(&mut s, &mut dict);

    let compressed = compress(&input, &dict);
    let mut file_contents = tree.serialize();
    dbg!(&file_contents);
    file_contents.extend_from_slice(&compressed.bf);

    let new_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(output_path)?;

    new_file.write_at(&file_contents, 0)?;

    Ok(())
}

fn do_decompression(input_path: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
    let input = fs::read(input_path)?;

    let tree = match HuffmanTree::from_buffer(input) {
        Some(tree) => tree,
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid file",
            )))
        }
    };

    println!("Tree constructed {:?}", tree);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Usage: {} <operation> <input-file> <output-file>", args[0]);
        return Ok(());
    }

    let operation = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    match operation.as_str() {
        "-c" => do_compression(input_path, output_path),
        "-d" => do_decompression(input_path, output_path),
        _ => {
            println!("Unknown command!");
            Ok(())
        }
    }
}

#[cfg(test)]
mod huffman_test {
    use crate::{
        compress, construct_huffman_tree, decompress, enqueue_chars, Bitfield, HuffmanTree, Prob,
    };
    use std::collections::HashMap;
    use std::rc::Rc;

    fn get_compressed(sut: &str) -> crate::Bitfield {
        let queue = enqueue_chars(sut);
        let tree = construct_huffman_tree(queue);
        let mut dict = HashMap::new();
        let mut stack = Vec::new();
        tree.construct_dictionary(&mut stack, &mut dict);
        compress(sut, &dict)
    }

    fn get_compressed_and_tree(sut: &str) -> (crate::Bitfield, crate::HuffmanTree) {
        let queue = enqueue_chars(sut);
        let tree = construct_huffman_tree(queue);
        let mut dict = HashMap::new();
        let mut stack = Vec::new();
        tree.construct_dictionary(&mut stack, &mut dict);
        for c in sut.chars() {
            let code = dict.get(&c).unwrap();
            println!("char {} has code {:#08b} with len {}", c, code.0, code.1);
        }

        (compress(sut, &dict), tree)
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
    fn test_bit_iteration() {
        let bits = Bitfield {
            bf: vec![1],
            len: 8,
            pos: 0,
        };

        let result: Vec<u8> = bits.into_iter().collect();
        assert_eq!(result, vec![0, 0, 0, 0, 0, 0, 0, 1]);

        let bits = Bitfield {
            bf: vec![4],
            len: 6,
            pos: 0,
        };

        let result: Vec<u8> = bits.into_iter().collect();
        assert_eq!(result, vec![0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_decompression() {
        let sut = "A";
        let (mut compressed, tree) = get_compressed_and_tree(sut);
        assert_eq!(decompress(&mut compressed, &tree), sut);

        let sut = "AAAAAAA";
        let (mut compressed, tree) = get_compressed_and_tree(sut);
        assert_eq!(decompress(&mut compressed, &tree), sut);

        let sut = "AB";
        let (mut compressed, tree) = get_compressed_and_tree(sut);
        assert_eq!(decompress(&mut compressed, &tree), sut);

        let sut = "ABC";
        let (mut compressed, tree) = get_compressed_and_tree(sut);
        assert_eq!(decompress(&mut compressed, &tree), sut);

        let sut = "A_DEAD_DAD_CEDED_A_BAD_BABE_A_BEADED_ABACA_";
        let (mut compressed, tree) = get_compressed_and_tree(sut);
        assert_eq!(decompress(&mut compressed, &tree), sut);
    }

    #[test]
    fn test_tree_serialization() {
        let tree: HuffmanTree = HuffmanTree::Node(
            Prob(0.0),
            Rc::new(HuffmanTree::Leaf('C', Prob(0.0))),
            Rc::new(HuffmanTree::Node(
                Prob(0.0),
                Rc::new(HuffmanTree::Leaf('A', Prob(0.0))),
                Rc::new(HuffmanTree::Leaf('B', Prob(0.0))),
            )),
        );

        let serialized = tree.serialize();
        assert_eq!(
            serialized,
            vec![5, HuffmanTree::NODE, b'C', HuffmanTree::NODE, b'A', b'B',]
        );

        let newtree = HuffmanTree::from_buffer(serialized).expect("Expected tree");
        assert_eq!(tree, *newtree.clone());
    }
}
