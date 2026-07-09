const BASE = '';

async function request(path, options = {}) {
  const url = BASE + path;
  const { body, formData, ...rest } = options;

  const fetchOpts = {
    headers: formData ? {} : { 'Content-Type': 'application/json' },
    ...rest,
  };

  if (body && !formData) {
    fetchOpts.body = JSON.stringify(body);
  } else if (formData) {
    fetchOpts.body = formData;
  }

  const res = await fetch(url, fetchOpts);

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch {
      // use default
    }
    throw new Error(detail);
  }

  return res;
}

export function get(path) {
  return request(path, { method: 'GET' }).then((r) => r.json());
}

export function post(path, body) {
  return request(path, { method: 'POST', body }).then((r) => r.json());
}

export function put(path, body) {
  return request(path, { method: 'PUT', body }).then((r) => r.json());
}

export function postForm(path, formData) {
  return request(path, { method: 'POST', formData }).then((r) => r.json());
}

export function del(path) {
  return request(path, { method: 'DELETE' }).then((r) => r.json());
}

export function postStream(path, body) {
  return request(path, { method: 'POST', body });
}
