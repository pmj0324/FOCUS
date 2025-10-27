# GitHub에 올리기

## 1단계: GitHub에서 저장소 만들기

1. https://github.com/new 방문
2. Repository name: `COSMO` 입력
3. Description: `Conditional Diffusion Model for 2D Dark Matter Maps` (선택사항)
4. **Public** 또는 **Private** 선택
5. ⚠️ **"Initialize this repository with"는 모두 체크 해제!** (README, .gitignore, license 등)
6. "Create repository" 클릭

## 2단계: 원격 저장소 연결 및 Push

GitHub에서 저장소를 만들면 다음 명령어들이 표시됩니다:

```bash
cd /Users/pmj0324/Sicence/cosmo/New

# 원격 저장소 추가
git remote add origin https://github.com/pmj0324/COSMO.git

# 브랜치 이름 확인 (main으로 설정되어 있음)
git branch -M main

# Push
git push -u origin main
```

## 완료!

이제 https://github.com/pmj0324/COSMO 에서 코드를 볼 수 있습니다.

## 주의사항

- 이미 같은 이름의 저장소가 있다면 다른 이름을 사용하거나 기존 저장소를 삭제하세요
- Private로 만들면 본인만 볼 수 있습니다
- Public으로 만들면 누구나 볼 수 있습니다 (데이터 파일은 .gitignore로 제외됨)

