#!/usr/bin/env python3
"""
Master test runner for all RAG system tests and diagnostics.
"""

import sys
import os
import subprocess
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_python_file(file_path: str, description: str) -> bool:
    """Run a Python file and return success status"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print('='*60)
    
    try:
        # Run the Python file
        result = subprocess.run([sys.executable, file_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(file_path))
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Check return code
        success = result.returncode == 0
        if success:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
        
        return success
        
    except Exception as e:
        print(f"💥 Error running {description}: {e}")
        traceback.print_exc()
        return False


def run_test_suite():
    """Run complete test suite"""
    print("🚀 RAG System Test Suite")
    print("="*60)
    print("This will run comprehensive tests to identify why queries are failing")
    print("="*60)
    
    # Get test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define tests to run in order
    tests = [
        ("debug_rag_system.py", "System Health Check & Diagnostics"),
        ("test_course_search_tool.py", "CourseSearchTool Tests"),
        ("test_ai_generator.py", "AI Generator Tests"),
        ("test_rag_integration.py", "RAG System Integration Tests"),
    ]
    
    results = []
    
    for test_file, description in tests:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            success = run_python_file(test_path, description)
            results.append((description, success))
        else:
            print(f"⚠️  Test file not found: {test_path}")
            results.append((description, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUITE SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    # Analysis and recommendations
    print(f"\n{'='*60}")
    print("💡 ANALYSIS & RECOMMENDATIONS")
    print('='*60)
    
    if passed == total:
        print("🎉 All tests passed! The system should be working correctly.")
        print("If you're still seeing 'query failed' errors, the issue may be:")
        print("  • Frontend-backend communication problems")
        print("  • Runtime environment differences")
        print("  • Real-time API connectivity issues")
    else:
        print("🔍 Some tests failed. Check the detailed output above for:")
        print("  • Missing or invalid API keys")
        print("  • Empty vector database (no courses loaded)")
        print("  • Component initialization failures") 
        print("  • Tool registration or execution issues")
        
        # Check if diagnostics ran successfully
        diagnostics_passed = any(desc == "System Health Check & Diagnostics" and success 
                               for desc, success in results)
        
        if diagnostics_passed:
            print("\n✅ Diagnostics completed - check the detailed health check above")
            print("   The health check should have identified the root cause")
        else:
            print("\n❌ Diagnostics failed - there may be fundamental setup issues")
            print("   Check Python imports and basic system configuration")
    
    return passed == total


def main():
    """Main test runner"""
    success = run_test_suite()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())