import Foundation

struct MNISTSample: Hashable {
    let label: UInt8
    let image: Data
}

extension MNISTSample {
    var target: [Float] {
        precondition(label < 10)
        let prefix = Array(repeating: Float.zero, count: Int(label))
        let value = [Float(1.0)]
        let postfix = Array(repeating: Float.zero, count: Int(9 - label))
        return prefix + value + postfix
    }
}
