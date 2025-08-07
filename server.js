const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const { exec, spawn } = require('child_process'); // ✅ you missed 'spawn'
const fs = require('fs');

const app = express();
const port = 8000;

// ✅ Enable CORS for your Vercel frontend and local dev
app.use(cors({
  origin: ['https://policy-reader.vercel.app', 'http://localhost:3000'],
  methods: ['GET', 'POST'],
  credentials: true,
}));

app.use(express.json());
app.use(express.static("public"));

// ✅ File Upload Configuration
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + file.originalname;
    cb(null, uniqueSuffix);
  },
});
const upload = multer({ storage });

// ✅ Upload PDF and Run index_faiss.py
app.post("/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const filePath = `uploads/${req.file.filename}`;
  console.log("✅ Uploaded:", req.file.filename);

  // Run index_faiss.py
  exec(`python3 projects/index_faiss.py ${filePath}`, (error, stdout, stderr) => {
    if (error) {
      console.error("❌ Error while building vector store:", stderr);
      return res.status(500).json({ error: "Vector store creation failed" });
    }
    console.log("✅ Vector store created");
    res.json({ success: true });
  });
});

// ✅ Handle Question & Run LLaMA
app.post("/ask", (req, res) => {
  const question = req.body.question;
  if (!question) {
    return res.status(400).json({ error: "Question is required" });
  }

  console.log("🔍 Question:", question);

  const py = spawn("python3", ["run_llama.py", question]);

  let output = "";
  py.stdout.on("data", (data) => {
    output += data.toString();
  });

  py.stderr.on("data", (data) => {
    console.error(`❌ Python error: ${data}`);
  });

  py.on("close", () => {
    console.log("✅ LLaMA Response:", output.trim());
    res.json({ answer: output.trim() });
  });
});

// ✅ Start the Server
app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
