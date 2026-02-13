function ExtractReportDate(raw_value: string, hint: string) -> string {
  client CustomGPT5Mini
  prompt #"
    Extract a date or period label from the following text.

    Hint: {{ hint }}

    Text: {{ raw_value }}

    Return ONLY the extracted value as a plain string.
    If nothing matches, return empty string.
  "#
}
