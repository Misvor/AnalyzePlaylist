// htmx json-enc extension (vendored from htmx-extensions v2.x).
// Source: https://github.com/bigskysoftware/htmx-extensions/blob/main/src/json-enc/json-enc.js
// License: Zero-Clause BSD (https://github.com/bigskysoftware/htmx-extensions/blob/main/LICENSE)
//
// Usage: add hx-ext="json-enc" to any hx-post element. The extension
// rewrites the body as JSON instead of form-urlencoded, and sets the
// Content-Type header to application/json. This is the canonical way to
// POST a JSON body from htmx 2.x since hx-encoding only supports
// application/x-www-form-urlencoded and multipart/form-data.
htmx.defineExtension("json-enc", {
  onEvent: function (name, evt) {
    if (name === "htmx:configRequest") {
      evt.detail.headers["Content-Type"] = "application/json";
    }
  },
  encodeParameters: function (xhr, parameters, elt) {
    return JSON.stringify(parameters);
  },
});
