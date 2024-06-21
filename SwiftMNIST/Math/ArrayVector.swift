import Foundation
import simd

infix operator .+: AdditionPrecedence
infix operator .-: AdditionPrecedence
infix operator .*: MultiplicationPrecedence
infix operator .**: ExponentiativePrecedence

extension Array where Element: AdditiveArithmetic {
    @inline(__always)
    static func .+(lhs: Self, rhs: Self) -> Self {
//        zip(lhs, rhs).map { $0 + $1 }
        Array(unsafeUninitializedCapacity: lhs.count) { buffer, initializedCount in
            for index in lhs.indices {
                buffer[index] = lhs[index] + rhs[index]
            }
            initializedCount = lhs.count
        }
    }

    @inline(__always)
    static func .+(lhs: Self, rhs: Element) -> Self {
//        lhs.map { $0 + rhs }
        Array(unsafeUninitializedCapacity: lhs.count) { buffer, initializedCount in
            for index in lhs.indices {
                buffer[index] = lhs[index] + rhs
            }
            initializedCount = lhs.count
        }
    }

    @inline(__always)
    static func .-(lhs: Self, rhs: Self) -> Self {
//        zip(lhs, rhs).map { $0 - $1 }
        Array(unsafeUninitializedCapacity: lhs.count) { buffer, initializedCount in
            for index in lhs.indices {
                buffer[index] = lhs[index] - rhs[index]
            }
            initializedCount = lhs.count
        }
    }

    @inline(__always)
    static func .-(lhs: Self, rhs: Element) -> Self {
//        lhs.map { $0 - rhs }
        Array(unsafeUninitializedCapacity: lhs.count) { buffer, initializedCount in
            for index in lhs.indices {
                buffer[index] = lhs[index] - rhs
            }
            initializedCount = lhs.count
        }
    }
}

extension Array where Element: AdditiveArithmetic {
    @inline(__always)
    var sum: Element {
//        reduce(.zero) { $0 + $1 }
        var total = Element.zero
        for element in self {
            total += element
        }
        return total
    }
}

extension Array where Element == Float {
    @inline(__always)
    var sum: Element {
        if true {
            SIMDHelper.simdSum(self)
        } else {
            reduce(.zero) { $0 + $1 }
        }
    }
}

extension Array where Element: BinaryFloatingPoint {
    @inline(__always)
    static func .*(lhs: Self, rhs: Self) -> Self {
//        zip(lhs, rhs).map { $0 * $1 }
        Array(unsafeUninitializedCapacity: lhs.count) { buffer, initializedCount in
            for index in lhs.indices {
                buffer[index] = lhs[index] * rhs[index]
            }
            initializedCount = lhs.count
        }
    }

    @inline(__always)
    static func .*(lhs: Self, rhs: Element) -> Self {
//        lhs.map { $0 * rhs }
        Array(unsafeUninitializedCapacity: lhs.count) { buffer, initializedCount in
            for index in lhs.indices {
                buffer[index] = lhs[index] * rhs
            }
            initializedCount = lhs.count
        }
    }
}

extension Array where Element: BinaryFloatingPoint {
    @inline(__always)
    static func .**(lhs: Self, rhs: Self) -> Self {
        zip(lhs, rhs).map { a, b in a ** b }
    }

    @inline(__always)
    static func .**(lhs: Self, rhs: Double) -> Self {
        lhs.map { element in Element(Double(element) ** rhs) }
    }
}

extension Array where Element: BinaryFloatingPoint {
    @inline(__always)
    func mean() -> Element {
        guard !isEmpty else { return .zero }
        return sum / Element(count)
    }

    @inline(__always)
    var squared: Self {
        map { $0 * $0 }
    }

    @inline(__always)
    var normalized: Self {
        let length = sqrt(reduce(0) { $0 + $1 * $1 })
        guard length > 0 else {
            return Array(
                repeating: 1 / Element(count),
                count: count
            )
        }
        return map { n in n / length }

//        let sum = reduce(0) { $0 + $1 }
//        guard sum > 0 else {
//            return Array(
//                repeating: 1 / Element(count),
//                count: count
//            )
//        }
//        return map { n in n / sum }
    }

    @inline(__always)
    var variance: Element {
        let meanValue = mean()
        let squaredDifferences = map { ($0 - meanValue) * ($0 - meanValue) }
        return squaredDifferences.mean()
    }
}

@inline(__always)
func abs<E: SignedNumeric & Comparable>(_ elements: Array<E>) -> Array<E> {
    elements.map { abs($0) }
}

private struct SIMDHelper {
    @inline(__always)
    static func simdSum(_ input: Array<Float>) -> Float {
        typealias SIMD = SIMD16<Float>

        var result = input
        while result.count > SIMD.scalarCount {
            result = result.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) in
                let elementCount = result.count
                var index = 0
                var batch = [Float]()
                batch.reserveCapacity(result.count / SIMD.scalarCount)

                while index + SIMD.scalarCount < elementCount {
                    let byteOffset = index * MemoryLayout<Float>.size
                    let casted = pointer.load(fromByteOffset: byteOffset, as: SIMD.self)
                    batch.append(simd_reduce_add(casted))
                    index += SIMD.scalarCount
                }

                return batch + result[index...]
            }
        }

        if result.count == 1 {
            return result[0]
        }

        var last = SIMD(repeating: 0)
        for (index, item) in result.enumerated() {
            last[index] = item
        }

        return simd_reduce_add(last)
    }
}
