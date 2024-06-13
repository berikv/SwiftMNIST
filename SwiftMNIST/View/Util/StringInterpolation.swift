import Foundation

extension String.StringInterpolation {
    static let formatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter
    }()

    /// Show the relative time changed between two dates
    mutating func appendInterpolation(relative range: ClosedRange<Date>) {
        let formattedString = Self.formatter.localizedString(for: range.upperBound, relativeTo: range.lowerBound)
        appendLiteral(formattedString)
    }

    // This is needed because the custom string interpolation above overrides the
    // existing implementation.. Something weird in Swift..
    mutating func appendInterpolation(_ value: Double, specifier: String) {
        let formattedString = String(format: specifier, value)
        appendLiteral(formattedString)
    }
}
