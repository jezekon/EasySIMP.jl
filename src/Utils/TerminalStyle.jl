# Print a message in the given ANSI style. Leading newlines of the message are printed
# before the styled part, so that `print_info("\nStep 2")` leaves an empty line.
function print_styled(msg::String, style::String, label::String)
    # Count leading newlines
    leading_newlines = length(match(r"^\n*", msg).match)
    # Remove leading newlines and get clean message
    cleaned_msg = lstrip(msg, '\n')

    # Print the leading newlines first if any exist
    if leading_newlines > 0
        print("\n" ^ leading_newlines)
    end

    # Print the formatted message
    println("\e[$(style)m$(label) $(cleaned_msg)\e[0m")
end

# Info messages - blue color for general information
print_info(msg::String) = print_styled(msg, "1;34", "[INFO]")

# Error messages - red color for critical issues
print_error(msg::String) = print_styled(msg, "1;31", "[ERROR]")

# Warning messages - orange/yellow color for potential issues
print_warning(msg::String) = print_styled(msg, "1;33", "[WARNING]")

# Success messages - green color for completion and positive results
print_success(msg::String) = print_styled(msg, "1;32", "[SUCCESS]")

# Highlighting important messages
print_data(msg::String) = print_styled(msg, "33", "")
