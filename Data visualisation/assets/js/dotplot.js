var dotplot = {}
dotplot.init = function() {
    
    // set the dimensions and margins of the graph
    var margin = {top: 30, right: 30, bottom: 20, left: 100},
        width = 500 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#dot-plot")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    // Parse the Data
    d3.csv("data/merged_df.csv", function(data) {

        // Filter data for year 2010
        data = data.filter(function(d) {
            var year = new Date(d.Year).getFullYear();
            return year === 2010;
        });

        // Find the maximum value across all variables
        var maxValue = d3.max(data, function(d) {
            return d3.max([d.Calories, d.Fat, d.Protein, d.Fruit, d.Sugar, d.Veg]);
        });
        

        // Add X axis
        var x = d3.scaleLinear()
            .domain([0, 300])
            .range([0, width]);
        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))

        // Y axis
        var y = d3.scaleBand()
            .range([ 0, height ])
            .domain(data.map(function(d) { return d.Country; }))
            .padding(1);
        svg.append("g")
            .call(d3.axisLeft(y))

        // Lines
        svg.selectAll("myline")
        .data(data)
        .enter()
        .append("line")
            .attr("x1", function(d) { return x(d3.min([d.Fat, d.Protein, d.Fruit, d.Sugar, d.Veg])); })
            .attr("x2", function(d) { return x(d3.max([d.Fat, d.Protein, d.Fruit, d.Sugar, d.Veg])); })
            .attr("y1", function(d) { return y(d.Country); })
            .attr("y2", function(d) { return y(d.Country); })
            .attr("stroke", "grey")
            .attr("stroke-width", "1px")
        

        // Circles of variable 1
        /* svg.selectAll("mycircle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", function(d) { return x(d.Calories); })
            .attr("cy", function(d) { return y(d.Country); })
            .attr("r", "3")
            .style("fill", "#f03b20")
        */

        // Circles of variable 2
        svg.selectAll("mycircle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", function(d) { return x(d.Fat); })
            .attr("cy", function(d) { return y(d.Country); })
            .attr("r", "3")
            .style("fill", "#ffffcc")
         
        // Circles of variable 3
         svg.selectAll("mycircle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", function(d) { return x(d.Protein); })
            .attr("cy", function(d) { return y(d.Country); })
            .attr("r", "3")
            .style("fill", "#a1dab4")
        
        // Circles of variable 4
        svg.selectAll("mycircle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", function(d) { return x(d.Fruit); })
            .attr("cy", function(d) { return y(d.Country); })
            .attr("r", "3")
            .style("fill", "#41b6c4")
        
        // Circles of variable 5
        svg.selectAll("mycircle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", function(d) { return x(d.Sugar); })
            .attr("cy", function(d) { return y(d.Country); })
            .attr("r", "3")
            .style("fill", "#2c7fb8")
        
        // Circles of variable 6
        svg.selectAll("mycircle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", function(d) { return x(d.Veg); })
            .attr("cy", function(d) { return y(d.Country); })
            .attr("r", "3")
            .style("fill", "#253494")

        // Add legend
        var legendGroup = svg.append("g")
          .attr("class", "legend")
          .attr("transform", "translate(" + (margin.left + 200) + "," + (height + margin.top - 400) + ")");
    
        var legend = legendGroup.selectAll(".legend-item")
          .data([
            // { label: "Calories", color: "#f03b20" },
            { label: "Fat", color: "#ffffcc" },
            { label: "Protein", color: "#a1dab4" },
            { label: "Fruit", color: "#41b6c4" },
            { label: "Sugar", color: "#2c7fb8" },
            { label: "Veg", color: "#253494" }
          ])
          .enter().append("g")
          .attr("class", "legend-item")
          .attr("transform", function(d, i) { return "translate(0," + (i * 20 + 20) + ")"; });
    
        legend.append("rect")
          .attr("x", 0)
          .attr("width", 18)
          .attr("height", 18)
          .style("fill", function(d) { return d.color; });
    
        legend.append("text")
          .attr("x", 25)
          .attr("y", 9)
          .attr("dy", ".35em")
          .style("text-anchor", "start")
          .text(function(d) { return d.label; });
    })

}