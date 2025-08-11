const ENDPOINTS = {
  single: 'http://127.0.0.1:5000/api/predict-url',
  multi: 'http://127.0.0.1:5000/api/predict-multi-model'
};

const el = id => document.getElementById(id);

async function fillCurrentTabUrl(){
  try {
    const [tab] = await chrome.tabs.query({active:true,currentWindow:true});
    if(tab && tab.url) el('url').value = tab.url;
  } catch (err) {
    console.error(err);
  }
}

function syntaxHighlight(json) {
  if (typeof json != 'string') {
    json = JSON.stringify(json, undefined, 2);
  }
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
    let cls = 'number';
    if (/^"/.test(match)) {
      if (/:$/.test(match)) {
        cls = 'key';
      } else {
        cls = 'string';
      }
    } else if (/true|false/.test(match)) {
      cls = 'boolean';
    } else if (/null/.test(match)) {
      cls = 'null';
    }
    return '<span class="' + cls + '">' + match + '</span>';
  });
}

function buildPayload(){
  const url = el('url').value.trim();
  const threshold = parseFloat(el('threshold').value);
  const endpointType = el('endpoint').value;
  if(endpointType === 'multi'){
    return { url, threshold };
  }
  const model = el('model').value;
  const numerical = el('numerical').checked ? 1 : 0;
  return { url, model, threshold, numerical };
}

function setStatus(text){ el('status').textContent = text; }
function showResult(obj){ el('result').innerHTML = syntaxHighlight(obj); }

async function send(){
  const endpointType = el('endpoint').value;
  const payload = buildPayload();
  const API_ENDPOINT = ENDPOINTS[endpointType];
  setStatus('Sending...');
  try{
    const resp = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if(!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    setStatus('OK');
    showResult(data);
  }catch(err){
    setStatus('Error: ' + err.message);
    console.error('Fetch error', err);
    showResult({ error: err.message });
  }
}

el('endpoint').addEventListener('change', () => {
  const isMulti = el('endpoint').value === 'multi';
  document.getElementById('label-model').style.display = isMulti ? 'none' : 'block';
  document.getElementById('label-numerical').style.display = isMulti ? 'none' : 'block';
});

el('send').addEventListener('click', send);
el('fill-current').addEventListener('click', fillCurrentTabUrl);
fillCurrentTabUrl();