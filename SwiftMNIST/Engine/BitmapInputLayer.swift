import Foundation

struct BitmapInputLayer: InputLayerType {
    typealias SIMD = SIMD16<Float>

    func forward(input: Data) -> [SIMD] {
        [SIMD](packing: input.map { Float($0) / Float(UInt8.max) })
    }
}
