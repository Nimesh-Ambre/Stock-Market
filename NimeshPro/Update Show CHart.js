<script>
    function updateChart() {
        const symbol = document.getElementById('symbolSelect').value;
        window.location.href = `/dashboard?symbol=${symbol}`;
    }
</script>
