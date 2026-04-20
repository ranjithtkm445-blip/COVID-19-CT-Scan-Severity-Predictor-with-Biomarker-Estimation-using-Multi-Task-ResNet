// File    : script.js
// Purpose : Handle file dropdown loading, analyze button click,
//           display CT slice, GradCAM, severity prediction,
//           biomarker results and PDF download link.

const SEV_COLORS = ['#2ECC71', '#3498DB', '#F1C40F', '#E67E22', '#E74C3C'];
const SEV_LABELS = [
    'No Findings',
    'Minimal (less than 25%)',
    'Moderate (25 to 50%)',
    'Significant (50 to 75%)',
    'Critical (more than 75%)'
];

const sel         = document.getElementById('fileSelect');
const analyzeBtn  = document.getElementById('analyzeBtn');
const loading     = document.getElementById('loading');
const placeholder = document.getElementById('placeholder');
const results     = document.getElementById('results');


// Load .nii files from upload folder
async function loadFiles() {
    try {
        const res  = await fetch('/folder_files');
        const data = await res.json();
        sel.innerHTML = '';

        if (data.files.length === 0) {
            sel.innerHTML = '<option value="">-- No .nii files in upload folder --</option>';
            analyzeBtn.disabled = true;
        } else {
            sel.innerHTML = '<option value="">-- Select a CT scan --</option>' +
                data.files.map(f => `<option value="${f}">${f}</option>`).join('');
            analyzeBtn.disabled = true;
        }
    } catch (e) {
        sel.innerHTML = '<option value="">-- Error loading files --</option>';
    }
}


// Enable analyze button when file selected
sel.addEventListener('change', () => {
    analyzeBtn.disabled = sel.value === '';
});


// Run prediction
async function analyze() {
    const filename = sel.value;
    if (!filename) return;

    placeholder.style.display = 'none';
    results.style.display     = 'none';
    loading.style.display     = 'block';
    analyzeBtn.disabled       = true;

    try {
        const res  = await fetch('/predict_folder', {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify({ filename })
        });
        const data = await res.json();

        if (data.error) {
            alert('Error: ' + data.error);
            loading.style.display = 'none';
            analyzeBtn.disabled   = false;
            return;
        }

        // CT + GradCAM images
        document.getElementById('ctImg').src  = 'data:image/png;base64,' + data.ct_image;
        document.getElementById('camImg').src = 'data:image/png;base64,' + data.cam_image;

        // Severity
        const sc = data.severity_class;
        const sevVal = document.getElementById('sevVal');
        sevVal.textContent = 'CT-' + sc;
        sevVal.style.color = SEV_COLORS[sc];
        document.getElementById('sevLbl').textContent = SEV_LABELS[sc];

        // Confidence bars
        document.getElementById('confBars').innerHTML =
            data.probabilities.map((p, i) => `
                <div class="bar-row">
                    <span class="bar-lbl">CT-${i}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:${(p * 100).toFixed(1)}%;background:${SEV_COLORS[i]}"></div>
                    </div>
                    <span class="bar-pct">${(p * 100).toFixed(0)}%</span>
                </div>`).join('');

        // Biomarkers
        const bm = data.biomarkers;
        const bmItems = [
            { name: 'CRP',     value: bm.CRP.toFixed(1),     unit: 'mg/L',  high: bm.CRP > 10 },
            { name: 'NLR',     value: bm.NLR.toFixed(1),     unit: 'ratio', high: bm.NLR > 3 },
            { name: 'D-dimer', value: bm.D_dimer.toFixed(2), unit: 'mg/L',  high: bm.D_dimer > 0.5 },
            { name: 'LDH',     value: Math.round(bm.LDH),    unit: 'U/L',   high: bm.LDH > 280 },
        ];

        document.getElementById('bmGrid').innerHTML = bmItems.map(b => `
            <div class="bm-item">
                <div class="bm-name">${b.name}</div>
                <div class="bm-val ${b.high ? 'bm-high' : 'bm-normal'}">${b.value}</div>
                <div class="bm-unit">${b.unit} &nbsp; ${b.high ? '&#9650; HIGH' : '&#10003; Normal'}</div>
            </div>`).join('');

        // PDF link
        document.getElementById('pdfLink').href = '/download/' + data.pdf_name;

        loading.style.display = 'none';
        results.style.display = 'block';
        analyzeBtn.disabled   = false;

    } catch (e) {
        alert('Request failed: ' + e);
        loading.style.display = 'none';
        analyzeBtn.disabled   = false;
    }
}


// Auto refresh every 15 seconds
setInterval(loadFiles, 15000);

// Load files on page start
loadFiles();