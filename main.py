"""
Agentic Deepfake Classifier - CLI Interface
"""

import argparse
import sys
import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
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
    
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='Path to the video file')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--sample-rate', '-s', type=float, default=1.0,
                        help='Frame sampling rate (fps)')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                        help='Maximum frames to analyze')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick check mode (5 frames)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    try:
        from src import DeepfakeAnalyzer
        
        analyzer = DeepfakeAnalyzer(
            weights_path=args.weights,
            sample_rate=args.sample_rate,
            use_cuda=args.cuda,
            max_frames=args.max_frames
        )
        
        if args.quick:
            result_summary = analyzer.quick_check(args.video)
            print(f"\n{result_summary}\n")
        else:
            result = analyzer.analyze(args.video, show_progress=not args.quiet)
            
            if args.quiet:
                print(result.short_summary)
            else:
                print(result)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                logger.info(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
