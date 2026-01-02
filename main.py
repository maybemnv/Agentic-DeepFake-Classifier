"""
Agentic Deepfake Classifier
Command-line interface for deepfake video analysis.

Usage:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --weights model/FF++_c23.pth
"""

import argparse
import sys
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_path():
    """Add project root to path for imports."""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic Deepfake Video Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --video test.mp4
    python main.py --video test.mp4 --quick
    python main.py --video test.mp4 --output results.json
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to model weights file (default: model/FF++_c23.pth)'
    )
    
    parser.add_argument(
        '--sample-rate', '-s',
        type=float,
        default=1.0,
        help='Frame sampling rate in fps (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-frames', '-m',
        type=int,
        default=None,
        help='Maximum number of frames to analyze (default: all)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick check mode (analyze only 5 frames)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file path for results'
    )
    
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Use CUDA GPU acceleration if available'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (just verdict and confidence)'
    )
    
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Verbose output with debug logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Setup path
    setup_path()
    
    # Check video exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    try:
        # Import analyzer
        from src.analyzer import DeepfakeAnalyzer
        
        # Initialize analyzer
        analyzer = DeepfakeAnalyzer(
            weights_path=args.weights,
            sample_rate=args.sample_rate,
            use_cuda=args.cuda,
            max_frames=args.max_frames
        )
        
        if args.quick:
            # Quick check mode
            result_summary = analyzer.quick_check(args.video)
            print(f"\n{result_summary}\n")
        else:
            # Full analysis
            result = analyzer.analyze(
                args.video,
                show_progress=not args.quiet
            )
            
            if args.quiet:
                # Minimal output
                print(f"{result.short_summary}")
            else:
                # Full output
                print(result)
            
            # Save to JSON if requested
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                logger.info(f"Results saved to: {output_path}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
