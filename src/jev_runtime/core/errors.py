class ServiceError(Exception):
    def __init__(self, status, message): self.status, self.message = status, message; super().__init__(message)
def object_fields(value, allowed, required=()):
    if not isinstance(value, dict) or set(value)-set(allowed) or set(required)-set(value): raise ValueError('invalid object fields; expected '+', '.join(allowed))
