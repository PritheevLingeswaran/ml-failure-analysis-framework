# Security notes

This is an internal evaluation platform. Still:

- Do not load untrusted slice expressions.
- Keep dataset access controlled (PII).
- Store outputs in access-controlled locations.
- Treat evaluation results as sensitive if they encode user-level examples.
- For production, add:
  - authn/authz
  - request logging redaction
  - rate limits
  - signed artifact storage
