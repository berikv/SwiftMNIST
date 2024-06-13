import Foundation

struct MNISTSample: Hashable {
    let label: UInt8
    let image: Data
}

extension MNISTSample {
    var target: [Double] {
        precondition(label < 10)
        let prefix = Array(repeating: 0.0, count: Int(label))
        let value = [1.0]
        let postfix = Array(repeating: 0.0, count: Int(9 - label))
        return prefix + value + postfix
    }
}
