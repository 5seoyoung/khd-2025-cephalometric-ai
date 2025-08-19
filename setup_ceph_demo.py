#!/usr/bin/env python3
"""
Cephalometric AI Demo Project Setup
데모 프로젝트 디렉토리 구조를 생성합니다.
"""

import os

# 프로젝트 디렉토리 구조
directories = [
    # 데이터 관련
    "data/sample_images",
    "data/clinical_standards", 
    "data/demo_results",
    
    # 소스 코드
    "src/core",
    "src/utils", 
    "src/demo",
    
    # 테스트
    "tests",
    
    # 설정 및 스크립트
    "scripts",
    "configs",
    
    # 문서
    "docs",
    
    # 로그
    "logs"
]

# 필수 파일 목록
files_to_create = [
    # Python 패키지 초기화
    "src/__init__.py",
    "src/core/__init__.py", 
    "src/utils/__init__.py",
    "src/demo/__init__.py",
    "tests/__init__.py",
    
    # 설정 파일
    "requirements.txt",
    "README.md",
    ".gitignore",
    
    # 실행 스크립트
    "scripts/run_demo.sh",
    
    # 환경 설정
    ".env.example"
]

def create_project_structure():
    """프로젝트 디렉토리 구조 생성"""
    print("🚀 Cephalometric AI Demo Project 구조 생성 중...")
    
    # 디렉토리 생성
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # 파일 생성
    for file_path in files_to_create:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                if file_path.endswith('.py'):
                    f.write('# -*- coding: utf-8 -*-\n')
                elif file_path == 'README.md':
                    f.write('# Cephalometric AI - Demo Project\n\n측면두부규격방사선사진 AI 분석 데모\n')
            print(f"📄 Created: {file_path}")
    
    print("\n✅ 프로젝트 구조 생성 완료!")
    print("\n📋 다음 단계:")
    print("1. 기본 설정 파일들 구성")
    print("2. 데이터 파일 준비")
    print("3. 핵심 모듈 구현")
    print("4. 데모 UI 구현")

if __name__ == "__main__":
    create_project_structure()