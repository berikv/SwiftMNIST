import Foundation

infix operator .+
infix operator .-

extension Array where Element: AdditiveArithmetic {
    static func .+(lhs: Self, rhs: Self) -> Self {
        zip(lhs, rhs).map { a, b in a + b }
    }

    static func .-(lhs: Self, rhs: Self) -> Self {
        zip(lhs, rhs).map { a, b in a - b }
    }
}

extension Array where Element: AdditiveArithmetic {
    @inline(__always)
    var sum: Element {
        reduce(.zero) { $0 + $1 }
    }
}

infix operator .*: MultiplicationPrecedence

extension Array where Element: BinaryFloatingPoint {
    static func .*(lhs: Self, rhs: Self) -> Element {
        zip(lhs, rhs).map { a, b in a * b }.sum
    }
}

infix operator .**: ExponentiativePrecedence

extension Array where Element: BinaryFloatingPoint {
    static func .**(lhs: Self, rhs: Self) -> Self {
        zip(lhs, rhs).map { a, b in a ** b }
    }

    static func .**(lhs: Self, rhs: Double) -> Self {
        lhs.map { element in Element(Double(element) ** rhs) }
    }
}

extension Array where Element: BinaryFloatingPoint {
    @inline(__always)
    var mean: Element {
        guard !isEmpty else { return .zero }
        return sum / Element(count)
    }

    var normalized: Self {
        let sum = reduce(0) { $0 + $1 }
        guard sum > 0 else {
            return Array(
                repeating: 1 / Element(count),
                count: count
            )
        }
        return map { n in n / sum }
    }

    var variance: Element {
        let meanValue = mean
        let squaredDifferences = map { ($0 - meanValue) * ($0 - meanValue) }
        return squaredDifferences.mean
    }
}

func abs<E: SignedNumeric & Comparable>(_ elements: Array<E>) -> Array<E> {
    elements.map { abs($0) }
}
