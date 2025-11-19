function predictTrend() {
    const trend = document.getElementById("trendInput").value;
    if (!trend) {
        document.getElementById("predictionResult").innerText = "Please enter a trend name.";
        return;
    }

    fetch(`/predict?trend=${encodeURIComponent(trend)}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById("predictionResult").innerText = 
                `Predicted next week's score for "${trend}": ${data.predicted_score.toFixed(2)}`;
        })
        .catch(err => {
            console.error(err);
            document.getElementById("predictionResult").innerText = "Error predicting trend.";
        });
}
