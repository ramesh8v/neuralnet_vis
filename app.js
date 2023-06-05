const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

new Vue({
    el: '#app',
    data: {
        slope: 15,
        intercept: 2,
        dataSize: 1000,
        weight: 50,
        bias: 15,
        epochs: 50,
        loss: null,
        xs: null,
        ys: null,
        chart: null,
    },
    methods: {

        createChart() {
            const ctx = document.getElementById('myChart').getContext('2d');
            this.chart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Data Points',
                        data: [], // this will be filled in generateData
                        backgroundColor: 'rgba(0, 123, 255, 0.5)'
                    }, {
                        label: 'Regression Line',
                        data: [], // this will be filled in generateData
                        type: 'line',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        fill: false
                    }]
                },
                options: {
                    responsive: false,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom'
                        }
                    }
                }
            });
        },

        updateChart() {
            // update the regression line dataset
            const minMaxX = [Math.min(...this.xs.arraySync()), Math.max(...this.xs.arraySync())];
            this.chart.data.datasets[1].data = [
                {x: minMaxX[0], y: minMaxX[0] * this.weight + this.bias},
                {x: minMaxX[1], y: minMaxX[1] * this.weight + this.bias}
            ];
            this.chart.update();
        },

        async generateData() {
            const slope = parseFloat(this.slope);
            const intercept = parseFloat(this.intercept);
            const dataSize = parseInt(this.dataSize);
            const noise = tf.randomNormal([dataSize], mean=0, stdDev=0.5);
            this.xs = tf.randomUniform([dataSize]);
            this.ys = this.xs.mul(slope).add(intercept).add(noise);
            // after generating data, update the data points dataset
            const xsArray = this.xs.arraySync();
            const ysArray = this.ys.arraySync();
            this.chart.data.datasets[0].data = xsArray.map((x, i) => ({x: x, y: ysArray[i]}));
            this.updateChart(); // also update the regression line
        },
        async startTraining() {
            const weight = parseFloat(this.weight);
            const bias = parseFloat(this.bias);
            const epochs = parseInt(this.epochs);

            model.setWeights([
                tf.tensor2d([weight], [1, 1]),
                tf.tensor1d([bias])
            ]);            

            const history = [];
            for (let i = 0; i < epochs; i++) {
                const response = await model.fit(this.xs, this.ys, {epochs: 1, batchSize: 4});
                model.getWeights().forEach((weight, index) => {
                    if (index==0){
                        this.weight = weight.dataSync()[0];
                    } else {
                        this.bias = weight.dataSync()[0];
                    }
                });
                history.push({loss: response.history.loss[0]});
                this.loss = response.history.loss[0];
                this.updateChart();
            }  
        }
    },
    mounted() {
        this.createChart(); // create the chart when the Vue instance is mounted
    }
});
