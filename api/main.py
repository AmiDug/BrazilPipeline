import sys
import os
from pathlib import Path
import logging
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get absolute path to the project root
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to Python path")

# Add the pipeline directory to the path as well
pipeline_dir = os.path.join(parent_dir, "pipeline")
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)
    print(f"Added {pipeline_dir} to Python path")

# Import the FastAPI components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Olist Price Prediction Pipeline API",
    description="API for running the Olist e-commerce price prediction pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict():
    """Run the price prediction pipeline and return results"""
    try:
        # Dynamically load run_pipeline.py from the pipeline directory
        run_pipeline_path = os.path.join(pipeline_dir, "run_pipeline.py")

        if not os.path.exists(run_pipeline_path):
            logger.error(f"run_pipeline.py not found at {run_pipeline_path}")
            return {"status": "error", "message": f"run_pipeline.py not found at {run_pipeline_path}"}

        logger.info(f"Loading run_pipeline from: {run_pipeline_path}")

        # Load the run_pipeline module dynamically
        spec = importlib.util.spec_from_file_location("run_pipeline", run_pipeline_path)
        run_pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_pipeline_module)

        # Run the pipeline
        logger.info("Starting pipeline run")
        results = run_pipeline_module.run_pipeline()

        if results is None:
            logger.error("Pipeline returned None")
            return {"status": "error", "message": "Pipeline execution failed"}

        # Extract relevant metrics and information
        best_model = results["training_results"]["best_model_name"]
        metrics = results["evaluation_results"]["metrics"]

        return {
            "status": "success",
            "best_model": best_model,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        run_pipeline_path = os.path.join(pipeline_dir, "run_pipeline.py")
        pipeline_files = os.listdir(pipeline_dir) if os.path.exists(pipeline_dir) else []

        return {
            "status": "healthy",
            "python_path": sys.path[:3],
            "cwd": os.getcwd(),
            "pipeline_dir": pipeline_dir,
            "pipeline_exists": os.path.exists(pipeline_dir),
            "run_pipeline_exists": os.path.exists(run_pipeline_path),
            "pipeline_files": pipeline_files[:10]  # Show first 10 files
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "python_path": sys.path[:3],
            "cwd": os.getcwd()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)