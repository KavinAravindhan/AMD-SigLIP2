#!/usr/bin/env python3
"""
TFRecord File Inspector
Analyzes the structure and contents of TFRecord files
"""

import tensorflow as tf
import numpy as np
from collections import defaultdict
import argparse
import sys

def parse_tfrecord_fn(record):
    """Parse a single TFRecord example"""
    # Define the feature description based on common VQA patterns
    # This might need adjustment based on your specific format
    feature_description = {}
    
    # Try to parse as tf.train.Example first
    try:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        return example
    except:
        return None

def inspect_tfrecord(file_path, max_records=10):
    """
    Inspect TFRecord file and extract information about its structure
    """
    print(f"Inspecting TFRecord file: {file_path}")
    print("="*60)
    
    try:
        # Create dataset from TFRecord file
        dataset = tf.data.TFRecordDataset(file_path)
        
        record_count = 0
        feature_types = defaultdict(set)
        feature_shapes = defaultdict(set)
        sample_values = defaultdict(list)
        
        # Iterate through records
        for raw_record in dataset.take(max_records):
            record_count += 1
            print(f"\n--- Record {record_count} ---")
            
            # Parse the record
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            # Extract features
            features = example.features.feature
            
            for key, feature in features.items():
                # Determine feature type and value
                if feature.bytes_list.value:
                    feature_types[key].add('bytes')
                    value = feature.bytes_list.value[0]
                    feature_shapes[key].add(f"bytes_length: {len(value)}")
                    
                    # Try to decode as string
                    try:
                        decoded = value.decode('utf-8')
                        print(f"  {key} (bytes/string): {decoded[:100]}{'...' if len(decoded) > 100 else ''}")
                        if len(sample_values[key]) < 3:
                            sample_values[key].append(decoded)
                    except:
                        print(f"  {key} (bytes): {len(value)} bytes")
                        if len(sample_values[key]) < 3:
                            sample_values[key].append(f"<{len(value)} bytes>")
                
                elif feature.float_list.value:
                    feature_types[key].add('float')
                    values = list(feature.float_list.value)
                    feature_shapes[key].add(f"float_count: {len(values)}")
                    print(f"  {key} (float): {values[:5]}{'...' if len(values) > 5 else ''}")
                    if len(sample_values[key]) < 3:
                        sample_values[key].append(values[:10])  # Store first 10 values
                
                elif feature.int64_list.value:
                    feature_types[key].add('int64')
                    values = list(feature.int64_list.value)
                    feature_shapes[key].add(f"int64_count: {len(values)}")
                    print(f"  {key} (int64): {values[:5]}{'...' if len(values) > 5 else ''}")
                    if len(sample_values[key]) < 3:
                        sample_values[key].append(values[:10])  # Store first 10 values
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Records inspected: {record_count}")
        print(f"\nFeature Summary:")
        
        for key in sorted(feature_types.keys()):
            types = ', '.join(feature_types[key])
            shapes = ', '.join(feature_shapes[key])
            print(f"\n  Feature: {key}")
            print(f"    Type(s): {types}")
            print(f"    Shape(s): {shapes}")
            
            if sample_values[key]:
                print(f"    Sample values:")
                for i, sample in enumerate(sample_values[key][:3]):
                    print(f"      Sample {i+1}: {str(sample)[:100]}{'...' if len(str(sample)) > 100 else ''}")
        
        # Try to get total record count
        print(f"\nAttempting to count total records...")
        try:
            total_count = sum(1 for _ in dataset)
            print(f"Total records in file: {total_count}")
        except Exception as e:
            print(f"Could not count total records: {e}")
            
    except Exception as e:
        print(f"Error reading TFRecord file: {e}")
        print("\nTrying alternative parsing methods...")
        
        # Alternative: try reading as raw bytes and look for patterns
        try:
            with open(file_path, 'rb') as f:
                data = f.read(1024)  # Read first 1KB
                print(f"File size: {len(data)} bytes (first 1KB)")
                print(f"First 100 bytes (hex): {data[:100].hex()}")
                
                # Look for common patterns
                if b'image' in data:
                    print("Found 'image' keyword in data")
                if b'question' in data:
                    print("Found 'question' keyword in data")
                if b'answer' in data:
                    print("Found 'answer' keyword in data")
                    
        except Exception as e2:
            print(f"Could not read file as raw bytes: {e2}")

def main():
    parser = argparse.ArgumentParser(description='Inspect TFRecord file contents')
    parser.add_argument('file_path', help='Path to the TFRecord file')
    parser.add_argument('--max-records', type=int, default=10, 
                       help='Maximum number of records to inspect (default: 10)')
    
    args = parser.parse_args()
    
    inspect_tfrecord(args.file_path, args.max_records)

if __name__ == "__main__":
    # If running directly with hardcoded path
    file_path = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/VQA_v1.tfrecord"
    
    if len(sys.argv) > 1:
        main()
    else:
        inspect_tfrecord(file_path, max_records=10)