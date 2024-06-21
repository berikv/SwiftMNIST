import Foundation

struct BitmapInputLayer: InputLayerType {
    func forward(input: Data) -> [Float] {
        precondition(input.count == 28*28)
//        return input.map { Float($0) / (Float(UInt8.max) / 2) - 1 }
        return input.map { Float($0) / Float(UInt8.max) }
    }
}
