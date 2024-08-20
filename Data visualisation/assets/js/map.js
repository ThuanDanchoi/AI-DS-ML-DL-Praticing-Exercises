// Load the external data
Promise.all([
    d3.json('assets/json/map.json'), // Ensure this is the path to your TopoJSON file
    d3.csv('df.csv') // Ensure this is the path to your CSV data
]).then(function([mapData, csvData]) {
    const svg = d3.select('#map').append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .style('display', 'block');

    const width = document.getElementById('map').clientWidth;
    const height = document.getElementById('map').clientHeight;
    let projection = d3.geoMercator().fitSize([width, height], mapData);
    let path = d3.geoPath().projection(projection);

    // Tooltip setup
    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0)
        .style('position', 'absolute')
        .style('padding', '10px')
        .style('background', 'rgba(0, 0, 0, 0.8)')
        .style('color', 'white')
        .style('pointer-events', 'none');

    // Process CSV data
    let dataByCountryYear = {};
    csvData.forEach(d => {
        const key = `${d.Country}-${d.Year}`;
        dataByCountryYear[key] = d; // Store whole data row
    });

    // Define continent coordinates and scales for zooming
    const continents = {
        'All': { center: [0, 0], scale: 130 }, // Default view
        'Australia': { center: [134.491000, -25.732887], scale: 1200 },
        'Europe': { center: [15.2551, 54.5260], scale: 600 },
        'Asia': { center: [100.6197, 34.0479], scale: 470 },
        'North America': { center: [-101.0000, 41.0000], scale: 700 },
        'South America': { center: [-58.3816, -34.6037], scale: 450 },
        'Africa': { center: [21.7587, 8.7832], scale: 500 }
    };

    // Update map function
    function updateMap() {
        const selectedYear = document.querySelector('input[name="year"]:checked').value;
        const selectedDataType = document.querySelector('input[name="data"]:checked').value;
        const selectedContinent = document.querySelector('input[name="continent"]:checked').value;

        // Set projection based on selected continent
        const center = continents[selectedContinent].center;
        const scale = continents[selectedContinent].scale;
        projection = d3.geoMercator().scale(scale).center(center).translate([width / 2, height / 2]);
        path = d3.geoPath().projection(projection);

        svg.selectAll('.country')
            .data(mapData.features)
            .join('path')
            .attr('class', 'country')
            .attr('d', path)
            .attr('fill', d => {
                const data = dataByCountryYear[`${d.properties.name}-${selectedYear}`];
                return data ? getColor(data[selectedDataType], selectedDataType) : '#ccc';
            })
            .attr('stroke', 'white')
            .on('mouseover', function(event, d) {
                svg.selectAll('.country').style('opacity', 0.5); // Make all countries more transparent
                d3.select(this).style('opacity', 1).classed('active', true); // Highlight the hovered country

                const key = `${d.properties.name}-${selectedYear}`;
                const data = dataByCountryYear[key];
                if (data) {
                    tooltip.style('opacity', 1)
                        .html(`Country: ${d.properties.name}<br>Year: ${data.Year}<br>${selectedDataType}: ${data[selectedDataType]}`)
                        .style('left', `${event.pageX + 20}px`)
                        .style('top', `${event.pageY + 20}px`);
                }
            })
            .on('mouseout', function() {
                svg.selectAll('.country').style('opacity', 1); // Reset opacity when mouse leaves
                d3.select(this).classed('active', false);
                tooltip.style('opacity', 0);
            });

        // Refresh map
        svg.selectAll('.country')
            .transition()
            .attr('d', path);
    }

    // Display button event listener
    document.getElementById('display').addEventListener('click', updateMap);
});

// Function to get color based on data type and value
function getColor(value, type) {
    const scales = {
        'Fat': d3.scaleSequential(d3.interpolateBlues).domain([0, 200]),
        'Protein': d3.scaleSequential(d3.interpolateGreens).domain([0, 100]),
        'Fruit': d3.scaleSequential(d3.interpolateOranges).domain([0, 500]),
        'Sugar': d3.scaleSequential(d3.interpolatePurples).domain([0, 300]),
        'Veg': d3.scaleSequential(d3.interpolateReds).domain([0, 500]),
        'Obesity': d3.scaleSequential(d3.interpolateCool).domain([0, 40])
    };
    return scales[type](value);
}

