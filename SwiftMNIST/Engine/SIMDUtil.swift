import Foundation

extension Array where Element == SIMD16<Float> { // SIMDScalar
    typealias SIMD = SIMD16<Float>

    init<C: RandomAccessCollection>(packing input: C) where C.Element == Float {
        var index = input.startIndex

        self.init()
        reserveCapacity(input.count / SIMD.scalarCount)

        while index < input.endIndex {
            // let upperBound = input.index(index, offsetBy: SIMD.scalarCount, limitedBy: input.endIndex)
            var upperBound = index
            guard input.formIndex(&upperBound, offsetBy: SIMD.scalarCount, limitedBy: input.endIndex)
            else { fatalError() }

            //            let paddingCount = max(0, upperBound - input.endIndex)
            let paddingCount = input.distance(from: upperBound, to: input.endIndex) % SIMD.scalarCount

            // Using Array(repeating: Float.zero, ...) makes the Swift compiler go nuts
            // Explicitly typing the Element as Float works around that
            let zeroPadding = [Float](repeating: .zero, count: paddingCount)
            let segmentWithPadding = input[index..<upperBound] + zeroPadding

            let element = SIMD16<Float>(segmentWithPadding)
            append(element)

            index = upperBound
        }
    }
}
