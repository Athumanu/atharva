document.addEventListener("DOMContentLoaded", () => {
    const uploadBtn = document.getElementById("uploadBtn");
    const imageUpload = document.getElementById("imageUpload");
    const previewContainer = document.getElementById("previewContainer");
    const resultsDiv = document.getElementById("results");
  
    // Handle click on Upload button to open file picker
    uploadBtn.addEventListener("click", () => {
      imageUpload.click();
    });
  
    // Handle file selection
    imageUpload.addEventListener("change", async (event) => {
      const file = event.target.files[0];
      if (!file) return;
  
      // Show image preview
      previewContainer.innerHTML = "";
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.style.width = "100px";
      img.style.border = "2px solid #007bff";
      img.style.margin = "10px";
      previewContainer.appendChild(img);
  
      // Prepare form data
      const formData = new FormData();
      formData.append("file", file);
  
      // Upload to Flask server
      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData
        });
  
        const result = await response.json();
        resultsDiv.innerHTML = ""; // Clear previous results
  
        if (result.matches && result.matches.length > 0) {
          result.matches.forEach(([path, dist]) => {
            const matchImg = document.createElement("img");
            matchImg.src = path;
            matchImg.style.width = "80px";
            matchImg.style.margin = "5px";
            matchImg.title = `Distance: ${dist.toFixed(2)}`;
            resultsDiv.appendChild(matchImg);
          });
        } else {
          resultsDiv.innerHTML = `<p>${result.message || result.error}</p>`;
        }
      } catch (error) {
        console.error("Upload failed:", error);
        resultsDiv.innerHTML = `<p>Upload failed. See console for details.</p>`;
      }
    });
  });
  