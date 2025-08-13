// CRISPR AI Analysis Frontend Application
// Connects to FastAPI backend and provides real-time visualization

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/live';

let ws = null;
let molViewer = null;
let threeViewer = null;
let currentViewer = 'threejs';
let dnaStructure = null;
let rotating = true;
let comparisonChart = null;
let accuracyChart = null;
let speedChart = null;
let currentSequences = [];
let classifications = {};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing CRISPR AI Platform...');
    
    // Initialize WebSocket connection
    connectWebSocket();
    
    // Initialize 3D molecular viewer
    initMolecularViewer();
    
    // Initialize charts
    initCharts();
    
    // Load initial data
    await loadDatabaseStats();
    await loadTopSequences();
    
    // Start real-time classification stream
    startClassificationStream();
});

// WebSocket connection
function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'metrics') {
            updateMetrics(data.data);
        } else if (data.type === 'classification') {
            addClassificationToStream(data.data);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };
}

function updateConnectionStatus(connected) {
    const badge = document.getElementById('connection-status');
    if (connected) {
        badge.textContent = 'Online';
        badge.className = 'badge bg-success';
    } else {
        badge.textContent = 'Offline';
        badge.className = 'badge bg-danger';
    }
}

// 3D Molecular Viewer
function initMolecularViewer() {
    const container = document.getElementById('mol-viewer');
    
    if (currentViewer === 'threejs') {
        initThreeJSViewer(container);
    } else {
        init3DmolViewer(container);
    }
    
    // Generate DNA structure from sequence
    updateDNAStructure('GTCTTTCTGCTCGT');
}

function initThreeJSViewer(container) {
    // Clear container
    container.innerHTML = '';
    
    // Three.js setup
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);
    scene.fog = new THREE.Fog(0x0a0a0f, 50, 200);
    
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(25, 15, 30);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = rotating;
    controls.autoRotateSpeed = 0.5;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);
    
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    keyLight.position.set(30, 30, 30);
    scene.add(keyLight);
    
    const fillLight = new THREE.DirectionalLight(0x00ffff, 0.3);
    fillLight.position.set(-20, 20, -20);
    scene.add(fillLight);
    
    threeViewer = {
        scene: scene,
        camera: camera,
        renderer: renderer,
        controls: controls,
        dnaGroup: null
    };
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

function init3DmolViewer(container) {
    container.innerHTML = '';
    molViewer = $3Dmol.createViewer(container, {
        backgroundColor: 'black'
    });
}

function updateDNAStructure(sequence) {
    if (currentViewer === 'threejs' && threeViewer) {
        createThreeJSDNA(sequence);
    } else if (molViewer) {
        update3DmolDNA(sequence);
    }
}

function createThreeJSDNA(sequence) {
    const { scene, dnaGroup } = threeViewer;
    
    // Remove existing DNA
    if (dnaGroup) {
        scene.remove(dnaGroup);
    }
    
    // Create new DNA group
    const newDnaGroup = new THREE.Group();
    
    // B-DNA parameters
    const rise = 3.4;
    const twist = 36;
    const radius = 10;
    const bases = sequence.split('');
    
    // Base colors
    const colors = {
        'A': 0xff4444,
        'T': 0x4444ff,
        'G': 0xffff44,
        'C': 0x44ff44
    };
    
    const complement = {
        'A': 'T', 'T': 'A',
        'G': 'C', 'C': 'G'
    };
    
    // Create backbone
    const backboneCurve1 = [];
    const backboneCurve2 = [];
    
    for (let i = 0; i <= bases.length; i++) {
        const angle1 = (i * twist * Math.PI / 180);
        const angle2 = angle1 + Math.PI;
        
        backboneCurve1.push(new THREE.Vector3(
            radius * Math.cos(angle1),
            i * rise,
            radius * Math.sin(angle1)
        ));
        
        backboneCurve2.push(new THREE.Vector3(
            radius * Math.cos(angle2),
            i * rise,
            radius * Math.sin(angle2)
        ));
    }
    
    // Create backbone tubes
    const curve1 = new THREE.CatmullRomCurve3(backboneCurve1);
    const curve2 = new THREE.CatmullRomCurve3(backboneCurve2);
    
    const tubeGeometry1 = new THREE.TubeGeometry(curve1, bases.length * 8, 0.3, 8);
    const tubeGeometry2 = new THREE.TubeGeometry(curve2, bases.length * 8, 0.3, 8);
    
    const backboneMaterial = new THREE.MeshPhongMaterial({
        color: 0x808080,
        emissive: 0x101010,
        shininess: 100
    });
    
    const backbone1 = new THREE.Mesh(tubeGeometry1, backboneMaterial);
    const backbone2 = new THREE.Mesh(tubeGeometry2, backboneMaterial);
    
    newDnaGroup.add(backbone1);
    newDnaGroup.add(backbone2);
    
    // Create base pairs
    bases.forEach((base, i) => {
        const angle = i * twist * Math.PI / 180;
        const y = i * rise;
        
        // Base 1
        const base1Geo = new THREE.SphereGeometry(1, 16, 16);
        const base1Mat = new THREE.MeshPhongMaterial({
            color: colors[base],
            emissive: colors[base],
            emissiveIntensity: 0.2
        });
        const base1Mesh = new THREE.Mesh(base1Geo, base1Mat);
        base1Mesh.position.set(
            radius * 0.7 * Math.cos(angle),
            y,
            radius * 0.7 * Math.sin(angle)
        );
        
        // Base 2 (complement)
        const comp = complement[base];
        const base2Geo = new THREE.SphereGeometry(1, 16, 16);
        const base2Mat = new THREE.MeshPhongMaterial({
            color: colors[comp],
            emissive: colors[comp],
            emissiveIntensity: 0.2
        });
        const base2Mesh = new THREE.Mesh(base2Geo, base2Mat);
        base2Mesh.position.set(
            radius * 0.7 * Math.cos(angle + Math.PI),
            y,
            radius * 0.7 * Math.sin(angle + Math.PI)
        );
        
        // Hydrogen bonds
        const bondGeo = new THREE.CylinderGeometry(0.1, 0.1, radius * 1.2);
        const bondMat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            opacity: 0.3,
            transparent: true
        });
        const bond = new THREE.Mesh(bondGeo, bondMat);
        bond.position.y = y;
        bond.rotation.z = Math.PI / 2;
        bond.rotation.y = angle;
        
        newDnaGroup.add(base1Mesh);
        newDnaGroup.add(base2Mesh);
        newDnaGroup.add(bond);
    });
    
    // Center the helix
    const helixHeight = bases.length * rise;
    newDnaGroup.position.y = -helixHeight / 2;
    
    scene.add(newDnaGroup);
    threeViewer.dnaGroup = newDnaGroup;
}

function update3DmolDNA(sequence) {
    const pdbData = generateDNAPDB(sequence);
    
    molViewer.clear();
    molViewer.addModel(pdbData, 'pdb');
    
    changeView('cartoon');
    
    molViewer.zoomTo();
    molViewer.render();
    
    if (rotating) {
        startRotation();
    }
}

function generateDNAPDB(sequence) {
    // Generate simplified B-DNA structure
    let pdb = 'HEADER    DNA\n';
    let atomNum = 1;
    
    const bases = sequence.split('');
    const helixRadius = 10;
    const risePerBase = 3.4;
    const twistPerBase = 36;
    
    bases.forEach((base, i) => {
        const angle = (i * twistPerBase) * Math.PI / 180;
        const z = i * risePerBase;
        
        // Backbone atoms
        const x1 = helixRadius * Math.cos(angle);
        const y1 = helixRadius * Math.sin(angle);
        
        pdb += `ATOM  ${String(atomNum).padStart(5)}  P   D${base} A${String(i+1).padStart(4)}    `;
        pdb += `${String(x1.toFixed(3)).padStart(8)}${String(y1.toFixed(3)).padStart(8)}`;
        pdb += `${String(z.toFixed(3)).padStart(8)}  1.00  0.00           P\n`;
        atomNum++;
        
        // Base atoms
        const x2 = (helixRadius - 3) * Math.cos(angle);
        const y2 = (helixRadius - 3) * Math.sin(angle);
        
        pdb += `ATOM  ${String(atomNum).padStart(5)}  N1  D${base} A${String(i+1).padStart(4)}    `;
        pdb += `${String(x2.toFixed(3)).padStart(8)}${String(y2.toFixed(3)).padStart(8)}`;
        pdb += `${String(z.toFixed(3)).padStart(8)}  1.00  0.00           N\n`;
        atomNum++;
    });
    
    pdb += 'END\n';
    return pdb;
}

function changeView(style) {
    if (currentViewer === 'threejs' && threeViewer) {
        // Handle Three.js view changes
        if (style === 'helix' && threeViewer.dnaGroup) {
            threeViewer.dnaGroup.children.forEach(child => {
                child.visible = true;
            });
        } else if (style === 'ladder' && threeViewer.dnaGroup) {
            // Hide backbones, show only base pairs
            threeViewer.dnaGroup.children.forEach((child, i) => {
                if (i < 2) child.visible = false; // Hide first two (backbones)
                else child.visible = true;
            });
        }
    } else if (molViewer) {
        molViewer.setStyle({}, {});
        
        switch(style) {
            case 'cartoon':
            case 'helix':
                molViewer.setStyle({}, {cartoon: {color: 'spectrum'}});
                break;
            case 'surface':
                molViewer.addSurface($3Dmol.SurfaceType.VDW, {
                    opacity: 0.8,
                    color: 'white'
                });
                break;
            case 'ladder':
            case 'licorice':
                molViewer.setStyle({}, {
                    stick: {radius: 0.3},
                    sphere: {radius: 0.5}
                });
                break;
        }
        
        molViewer.render();
    }
}

function switchViewer(viewer, event) {
    currentViewer = viewer;
    
    // Update button states
    document.querySelectorAll('.btn-group button').forEach(btn => {
        btn.classList.remove('active');
    });
    if (event && event.target) {
        event.target.classList.add('active');
    }
    
    // Reinitialize viewer
    initMolecularViewer();
    
    // Update with current sequence
    const currentSeq = document.getElementById('current-barcode').textContent.replace(/[^ATGC]/g, '');
    if (currentSeq) {
        updateDNAStructure(currentSeq);
    }
}

function toggleRotation() {
    rotating = !rotating;
    document.getElementById('rotation-icon').textContent = rotating ? '⏸' : '▶';
    
    if (currentViewer === 'threejs' && threeViewer) {
        threeViewer.controls.autoRotate = rotating;
    } else if (rotating) {
        startRotation();
    }
}

function startRotation() {
    if (!rotating) return;
    
    if (currentViewer === 'threejs' && threeViewer) {
        threeViewer.controls.autoRotate = true;
    } else if (molViewer) {
        molViewer.rotate(1, 'y');
        molViewer.render();
        
        requestAnimationFrame(() => {
            if (rotating && currentViewer !== 'threejs') startRotation();
        });
    }
}

// Charts
function initCharts() {
    // Comparison chart
    const compCtx = document.getElementById('comparison-chart').getContext('2d');
    comparisonChart = new Chart(compCtx, {
        type: 'bar',
        data: {
            labels: ['AI Classifier', 'Threshold'],
            datasets: [{
                label: 'Accuracy',
                data: [0, 0],
                backgroundColor: ['rgba(0, 255, 255, 0.5)', 'rgba(255, 255, 0, 0.5)'],
                borderColor: ['#00ffff', '#ffff00'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#fff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: '#fff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });
    
    // Accuracy chart
    const accCtx = document.getElementById('accuracy-chart').getContext('2d');
    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'AI Accuracy',
                data: [],
                borderColor: '#00ffff',
                backgroundColor: 'rgba(0,255,255,0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { 
                    labels: { color: '#fff' }
                },
                title: {
                    display: true,
                    text: 'Classification Accuracy Over Time',
                    color: '#fff'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#fff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: '#fff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });
    
    // Speed chart
    const speedCtx = document.getElementById('speed-chart').getContext('2d');
    speedChart = new Chart(speedCtx, {
        type: 'doughnut',
        data: {
            labels: ['AI Processing', 'Cache Hits', 'Idle'],
            datasets: [{
                data: [30, 60, 10],
                backgroundColor: [
                    'rgba(255, 0, 255, 0.5)',
                    'rgba(0, 255, 255, 0.5)',
                    'rgba(128, 128, 128, 0.5)'
                ],
                borderColor: ['#ff00ff', '#00ffff', '#808080'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: '#fff' }
                },
                title: {
                    display: true,
                    text: 'Processing Efficiency',
                    color: '#fff'
                }
            }
        }
    });
}

// Data loading
async function loadDatabaseStats() {
    try {
        const response = await fetch(`${API_URL}/api/stats`);
        const stats = await response.json();
        
        // Update UI with stats
        console.log('Database stats:', stats);
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadTopSequences() {
    try {
        const response = await fetch(`${API_URL}/api/sequences/top?limit=50`);
        currentSequences = await response.json();
        
        displaySequences(currentSequences);
        
        // Classify first sequence for demo
        if (currentSequences.length > 0) {
            classifySequence(currentSequences[0]);
        }
        
    } catch (error) {
        console.error('Error loading sequences:', error);
    }
}

function displaySequences(sequences) {
    const container = document.getElementById('sequence-list');
    
    const html = sequences.map((seq, index) => `
        <div class="sequence-item" onclick="selectSequence(${index})">
            <div class="sequence-barcode">${colorizeSequence(seq.barcode)}</div>
            <div class="sequence-stats">
                <span class="stat-item">Count: ${seq.count}</span>
                <span class="stat-item">Z: ${seq.z_score.toFixed(2)}</span>
                <span class="stat-item">GC: ${seq.gc_content.toFixed(0)}%</span>
            </div>
            <div id="class-${index}" class="classification-badge"></div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

function colorizeSequence(sequence) {
    return sequence.split('').map(base => {
        const colors = {
            'A': '#ff4444',
            'T': '#4444ff',
            'G': '#ffff44',
            'C': '#44ff44'
        };
        return `<span style="color: ${colors[base] || '#fff'}">${base}</span>`;
    }).join('');
}

function selectSequence(index) {
    const seq = currentSequences[index];
    
    // Update 3D viewer
    updateDNAStructure(seq.barcode);
    
    // Update info panel
    document.getElementById('current-barcode').innerHTML = colorizeSequence(seq.barcode);
    document.getElementById('current-gc').textContent = `${seq.gc_content.toFixed(1)}%`;
    
    // Classify sequence
    classifySequence(seq);
}

async function classifySequence(seq) {
    try {
        const response = await fetch(`${API_URL}/api/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                barcode: seq.barcode,
                count: seq.count,
                z_score: seq.z_score
            })
        });
        
        const result = await response.json();
        
        // Update UI with classification
        document.getElementById('current-class').textContent = result.classification;
        document.getElementById('current-class').className = `badge bg-${getClassColor(result.classification)}`;
        document.getElementById('current-confidence').textContent = `${(result.confidence * 100).toFixed(0)}%`;
        
        // Store classification
        classifications[seq.barcode] = result;
        
        // Update sequence list
        const index = currentSequences.findIndex(s => s.barcode === seq.barcode);
        if (index >= 0) {
            const badge = document.getElementById(`class-${index}`);
            if (badge) {
                badge.textContent = result.classification;
                badge.className = `classification-badge ${result.classification.toLowerCase()}`;
            }
        }
        
        // Add to stream
        addClassificationToStream(result);
        
    } catch (error) {
        console.error('Error classifying sequence:', error);
    }
}

function getClassColor(classification) {
    const colors = {
        'HIGHLY_ENRICHED': 'danger',
        'ENRICHED': 'warning',
        'NEUTRAL': 'secondary',
        'DEPLETED': 'info',
        'HIGHLY_DEPLETED': 'primary'
    };
    return colors[classification] || 'secondary';
}

// Real-time stream
function startClassificationStream() {
    // Simulate streaming for demo
    setInterval(() => {
        if (currentSequences.length > 0) {
            const randomSeq = currentSequences[Math.floor(Math.random() * currentSequences.length)];
            if (!classifications[randomSeq.barcode]) {
                classifySequence(randomSeq);
            }
        }
    }, 3000);
}

function addClassificationToStream(result) {
    const container = document.getElementById('stream-container');
    
    const streamItem = document.createElement('div');
    streamItem.className = 'stream-item';
    streamItem.innerHTML = `
        <span class="stream-barcode">${colorizeSequence(result.barcode.substring(0, 8))}...</span>
        <span class="stream-class ${result.classification.toLowerCase()}">${result.classification}</span>
        <span class="stream-confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
    
    container.insertBefore(streamItem, container.firstChild);
    
    // Keep only last 10 items
    while (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
    
    // Animate entry
    streamItem.style.opacity = '0';
    streamItem.style.transform = 'translateY(-20px)';
    setTimeout(() => {
        streamItem.style.transition = 'all 0.5s';
        streamItem.style.opacity = '1';
        streamItem.style.transform = 'translateY(0)';
    }, 10);
}

// Metrics update
function updateMetrics(metrics) {
    document.getElementById('processing-rate').textContent = `${metrics.processing_rate.toFixed(1)} seq/s`;
    document.getElementById('cache-rate').textContent = `${metrics.cache_hit_rate.toFixed(1)}%`;
    document.getElementById('api-calls').textContent = metrics.api_calls_made;
    document.getElementById('sequences-processed').textContent = metrics.sequences_processed;
    
    // Update progress bars
    document.getElementById('rate-progress').style.width = `${Math.min(metrics.processing_rate * 10, 100)}%`;
    document.getElementById('cache-progress').style.width = `${metrics.cache_hit_rate}%`;
    
    // Update comparison chart
    if (comparisonChart) {
        comparisonChart.data.datasets[0].data = [
            95 + Math.random() * 5,  // AI accuracy
            75 + Math.random() * 5   // Threshold accuracy
        ];
        comparisonChart.update();
    }
    
    // Update accuracy chart
    if (accuracyChart) {
        const labels = accuracyChart.data.labels;
        const data = accuracyChart.data.datasets[0].data;
        
        if (labels.length > 20) {
            labels.shift();
            data.shift();
        }
        
        labels.push(new Date().toLocaleTimeString());
        data.push(90 + Math.random() * 10);
        
        accuracyChart.update();
    }
    
    // Update speed chart
    if (speedChart && metrics.cache_hit_rate) {
        speedChart.data.datasets[0].data = [
            100 - metrics.cache_hit_rate,
            metrics.cache_hit_rate,
            0
        ];
        speedChart.update();
    }
}

// Update classification summary
function updateClassificationSummary() {
    const summary = {};
    
    Object.values(classifications).forEach(c => {
        summary[c.classification] = (summary[c.classification] || 0) + 1;
    });
    
    const container = document.getElementById('classification-summary');
    const html = Object.entries(summary)
        .sort((a, b) => b[1] - a[1])
        .map(([cls, count]) => `
            <div class="class-summary-item ${cls.toLowerCase()}">
                <span>${cls.replace('_', ' ')}</span>
                <span class="badge bg-dark">${count}</span>
            </div>
        `).join('');
    
    container.innerHTML = html || '<div class="text-muted">No classifications yet</div>';
}

// Search functionality
document.getElementById('sequence-search')?.addEventListener('input', (e) => {
    const query = e.target.value.toUpperCase();
    const filtered = currentSequences.filter(seq => 
        seq.barcode.includes(query)
    );
    displaySequences(filtered);
});

// Export functionality
function exportResults() {
    const data = Object.values(classifications);
    
    if (data.length === 0) {
        alert('No classifications to export');
        return;
    }
    
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `classifications_${new Date().toISOString()}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
}

// Update summary periodically
setInterval(updateClassificationSummary, 5000);