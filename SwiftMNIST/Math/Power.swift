import Foundation

precedencegroup ExponentiativePrecedence {
  associativity: right
  higherThan: MultiplicationPrecedence
}

infix operator ** : ExponentiativePrecedence
infix operator **= : AssignmentPrecedence

/// Exponent operator.
public func ** <N: BinaryInteger>(base: N, power: N) -> N {
    return N.self(pow(Double(base), Double(power)))
}

/// Exponent operator.
public func ** <N: BinaryFloatingPoint>(base: N, power: N) -> N {
    return N.self(pow(Double(base), Double(power)))
}


public func **= <N: BinaryInteger>(lhs: inout N, rhs: N) {
    lhs = lhs ** rhs
}

public func **= <N: BinaryFloatingPoint>(lhs: inout N, rhs: N) {
    lhs = lhs ** rhs
}
