function showSection(sectionId) {
    // 1. Ocultar todas las secciones
    const sections = document.querySelectorAll('.web-section');
    sections.forEach(section => {
        section.classList.remove('visible');
        section.classList.add('hidden');
    });

    // 2. Mostrar la sección seleccionada
    const activeSection = document.getElementById(sectionId);
    if (activeSection) {
        // Quitamos 'hidden' y añadimos 'visible' para que CSS la muestre
        activeSection.classList.remove('hidden');
        activeSection.classList.add('visible');
    }
    
    // 3. Actualizar el estilo de los botones (opcional, pero elegante)
    const buttons = document.querySelectorAll('.navbar nav button');
    buttons.forEach(button => {
        button.classList.remove('active');
    });
    
    // El botón activo es el que tiene el ID correspondiente al de la sección
    const activeButton = document.getElementById('btn-' + sectionId);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

// Asegurar que la primera parte esté visible al cargar
document.addEventListener('DOMContentLoaded', () => {
    
    const convRSolarToREarth = 109.2; //Converison factor from Solar Radi to Earth radi
    const timeArray = Array(500).fill(1.0);
    let mouseX = 0, mouseY = 0;

    document.addEventListener('mousemove', (event) => {
        mouseX = event.clientX-canvas.getBoundingClientRect().left;
        mouseY = event.clientY-canvas.getBoundingClientRect().top;
    });


    function animateFrame(timestamp) {
        



        //Uptdate the light-graph
        const chartCanvas = document.getElementById('light-curve-chart');
        let fluxData = Chart.getChart(chartCanvas).data.datasets[0].data;
        
        const noiseSlider = document.getElementById('noise-checkbox');

        const starRadius =  convRSolarToREarth * parseFloat(starRadiusInput.value);
        const starArea = Math.PI*starRadius*starRadius;
        const bodyRadius = parseFloat(bodyRadiusInput.value);
        
        const starCenterY = canvas.height / 2;
        const starCenterX = canvas.width / 2;
        const distanceBodyToStar = Math.sqrt((mouseY - starCenterY)**2 + (mouseX-starCenterX)**2)
        
        const visibleArea = areaDifference(starRadius, bodyRadius, distanceBodyToStar);

        var error = 0

        if (noiseSlider.checked) { error = .0001*Math.sqrt( -2.0 * Math.log( Math.random() ) ) * Math.cos( 2.0 * Math.PI * Math.random()) }

        const flux = visibleArea/starArea + error
        fluxData  = getNewYValues(flux, fluxData);
        
        updateLightCurveChart(timeArray, fluxData);
        requestAnimationFrame(animateFrame);
    }





























    // Functions done by mi pana gemini



    const canvas = document.getElementById('simulation-canvas');
    const ctx = canvas.getContext('2d');

    // Input elements
    const starRadiusInput = document.getElementById('star-radius');
    const starMassInput = document.getElementById('star-mass');
    const starTempInput = document.getElementById('star-temp');
    const bodyRadiusInput = document.getElementById('body-radius');
    const orbitalPeriodInput = document.getElementById('orbital-period');

    const koiFeaturesDiv = document.getElementById('koi-features');
    let lightCurveChart;

    

    // --- Core Functions ---

    function setup() {
        setDefaultState();
        addEventListeners();
        requestAnimationFrame(animateFrame);
    }

    function setDefaultState() {
        const labels = Array.from({ length: 500 }, (_, i) => i);
        const flux = Array(500).fill(1.0);
        updateLightCurveChart(labels, flux, true);
        displayKoiFeatures(null); // Clear features
        clearCanvas();
        drawStar();
        
    }

    function resetState() {
        displayKoiFeatures(null); // Clear features
        clearCanvas();
        drawStar();
        
    }

    function handleInteraction(pos) {
        // Redraw simulation canvas
        clearCanvas();
        drawStar();
        drawPlanet(pos.x, pos.y);

        // Calculate parameters from inputs and mouse position
        const starRadius = parseFloat(starRadiusInput.value);
        const starMass = parseFloat(starMassInput.value);
        const starTemp = parseFloat(starTempInput.value);
        const bodyRadius = parseFloat(bodyRadiusInput.value);
        const orbitalPeriod = parseFloat(orbitalPeriodInput.value);
        
        const starRadiusPixels = starRadius * 50;
        const starCenterY = canvas.height / 2;
        const starCenterX = canvas.width / 2;
        const impactParam = Math.sqrt((pos.y - starCenterY)**2 + (pos.x-starCenterX)**2) / starRadiusPixels;
        
        
        // Calculate and display KOI features
        const features = calculateKoiFeaturesJS(starRadius, starMass, starTemp, bodyRadius, orbitalPeriod, impactParam);
        displayKoiFeatures(features);
    }

    // --- Event Listeners ---

    function addEventListeners() {
        canvas.addEventListener('mousemove', (e) => handleInteraction(getMousePos(e)));
        canvas.addEventListener('mouseleave', resetState);

    }

    // --- Drawing & UI ---

    function drawStar() {
        const starRadius = parseFloat(starRadiusInput.value) * convRSolarToREarth;
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, starRadius);
        gradient.addColorStop(0, 'rgba(255, 255, 220, 1)');
        gradient.addColorStop(0.8, 'rgba(255, 220, 180, 1)');
        gradient.addColorStop(1, 'rgba(255, 180, 100, 0.8)');
        ctx.beginPath();
        ctx.arc(centerX, centerY, starRadius, 0, 2 * Math.PI);
        ctx.fillStyle = gradient;
        ctx.fill();
    }

    function drawPlanet(x, y) {
        const bodyRadius = parseFloat(bodyRadiusInput.value);
        ctx.beginPath();
        ctx.arc(x, y, bodyRadius, 0, 2 * Math.PI);
        ctx.fillStyle = 'black';
        ctx.fill();
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
    }

    function updateLightCurveChart(labels, flux, forceNew = false) {
        const chartCanvas = document.getElementById('light-curve-chart');
        const newMin = flux ? Math.min(...flux) * 0.995 : 0.95;
        const newMax = flux ? Math.max(...flux) * 1.005 : 1.05;

        if (lightCurveChart && !forceNew) {
            lightCurveChart.data.labels = labels;
            lightCurveChart.data.datasets[0].data = flux;
            lightCurveChart.options.scales.y.min = newMin;
            lightCurveChart.options.scales.y.max = newMax;
            lightCurveChart.update('none');
        } else {
            if (lightCurveChart) lightCurveChart.destroy();
            lightCurveChart = new Chart(chartCanvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Flux',
                        data: flux,
                        borderColor: '#3498db',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    }]
                },
                options: {
                    animation: false,
                    scales: {
                        x: { title: { display: true, text: 'Transit Phase' }, ticks: { display: false }, grid: { display: false } },
                        y: { title: { display: true, text: 'Flux' }, min: newMin, max: newMax }
                    }
                }
            });
        }
    }

    function displayKoiFeatures(features) {
        if (!features) {
            koiFeaturesDiv.innerHTML = 'Hover the mouse over the star to simulate.';
            return;
        }
        let featuresHtml = '';
        for (const [key, value] of Object.entries(features)) {
            const formattedValue = (typeof value === 'number') ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : value;
            featuresHtml += `<strong>${key}:</strong> ${formattedValue}<br>`;
        }
        koiFeaturesDiv.innerHTML = featuresHtml;
    }

    // --- Initial Load ---
    setup();

    // Show loader while train/predict request is in progress
    const modelsForm = document.querySelector('#models form');
    const loaderEl = document.querySelector('#models .loader');
    const resultPre = document.getElementById('train-result');
    const trainBtn = document.getElementById('btn-train');
    const predictBtn = document.getElementById('btn-predict');
});