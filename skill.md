---
name: java-17-to-21
description: Assess a local Maven or Gradle Java 17 repository for Java 21 migration and validate proposed changes without automatically modifying code or configuration
allowed-tools: Bash, Read, Grep, Glob
---

# /java-17-to-21 Skill

Inspect the current local repository and produce an evidence-based Java 17 to Java 21 migration report. Follow the `java-17-to-21-baseline` instruction supplied by this pack.

## Usage

```text
/java-17-to-21
/java-17-to-21 validate
```

- Default mode performs read-only discovery and assessment.
- `validate` performs the same assessment, proposes validation commands, and runs them only after explicit user approval.

## Operating contract

- Work only in the current workspace unless the user explicitly identifies another local path.
- Do not edit files, install software, change dependencies, commit, push, deploy, or update external systems.
- Treat wrapper/build commands as execution of repository code. Never run Maven, Gradle, application, test, `jdeps`, or `jdeprscan` commands without first showing the exact commands and receiving explicit approval.
- File reads, searches, directory listing, `git status`, `git rev-parse`, and installed `java -version`/`javac -version` checks are read-only discovery and may run without approval.
- Redact secrets and sensitive configuration values from output.
- Mark facts that cannot be resolved as `UNRESOLVED`; never guess an effective value or dependency version.

## Workflow

### Step 1: Establish repository context

1. Determine the repository root with `git rev-parse --show-toplevel` when Git metadata is available; otherwise use the current directory and note that Git context is unavailable.
2. Record the current branch and dirty/clean state without changing it.
3. Find Java source and build files. Prefer `rg --files`; fall back to available file-search tools.
4. Detect Maven, Gradle, mixed builds, other build systems, and nested independent builds.
5. If no Java source/build is found, return `UNSUPPORTED`.
6. If the project is Android, Scala-only, Kotlin-only, Ant-only, Bazel-only, or otherwise outside the baseline, return `UNSUPPORTED` and explain why.

Do not assume the installed `java -version` is the repository's configured source/target version.

### Step 2: Discover Java and build configuration

Inspect all relevant modules and record file evidence.

**Maven:**

- `pom.xml` files, modules, parents, profiles, properties, plugin management, dependency management
- `maven.compiler.release`, compiler `<release>`, `<source>`, `<target>`, `java.version`
- Maven Wrapper properties and `.mvn/` configuration
- Maven Toolchains declarations
- Compiler, Surefire, Failsafe, Javadoc, JaCoCo, shading, packaging, and framework plugins

**Gradle:**

- `build.gradle`, `build.gradle.kts`, settings files, `gradle.properties`
- Java toolchains, `sourceCompatibility`, `targetCompatibility`, and `options.release`
- Root and subproject overrides, convention plugins, `buildSrc`, included builds
- Wrapper distribution version
- Version catalogs, platforms, plugin versions, and test launchers

**Delivery/runtime:**

- CI files, Jenkinsfiles, Dockerfiles, buildpacks, deployment manifests, startup scripts
- `.java-version`, `.sdkmanrc`, `.tool-versions`, and committed IDE compiler configuration
- Java/JVM environment variables and JVM flags, without exposing secret values

Report conflicting values and unresolved inheritance. Do not run Maven or Gradle merely to calculate effective configuration during assessment mode.

### Step 3: Analyze migration areas

Create findings in these categories:

1. Java compiler/release configuration
2. Maven/Gradle wrapper and build plugins
3. Dependencies, BOMs, platforms, and frameworks
4. Annotation processors and code generation
5. Test, coverage, bytecode, and agent tooling
6. Internal JDK APIs, reflection, and module-access flags
7. Removed/deprecated API and JVM-option indicators
8. Charset, locale, serialization, crypto/TLS, JNI/native, and behavioral risks
9. CI, container, packaging, and deployment runtime alignment
10. Missing or insufficient validation coverage

Use the pack baseline to label every finding `REQUIRED`, `ADVISORY`, or `UNRESOLVED`. Cite the repository file/property/line when available. A raw string match alone is not proof; explain what must confirm it.

For dependency compatibility:

- Prefer committed dependency-management evidence.
- Use authoritative vendor/project documentation if online lookup is available and permitted.
- Include the source and access date for an exact recommendation.
- Otherwise state that the version needs authoritative or organization-approved confirmation.

If Spring Boot or another framework migration is implicated, identify it as a follow-up. Do not expand this skill into a framework major-version migration.

### Step 4: Produce the assessment report

Print the report format below before proposing command execution:

```text
Java 17 -> 21 Migration Assessment

Overall status: <CHANGES REQUIRED | BLOCKED | PARTIALLY VALIDATED | UNSUPPORTED>
Mode: assessment | validation
Repository: <path>
Build: <Maven | Gradle | Mixed> <wrapper version or unresolved>
Configured Java: <value(s) with evidence>
Installed Java: <value or not checked>

Configuration Evidence
- <file:line/property -> detected value>

Required Changes
R1. <title>
    Evidence: <file:line/property/command>
    Why: <Java 21 impact>
    Recommendation: <specific change or decision>
    Validate: <check>

Advisory Recommendations
A1. <same structure>

Unresolved / High Risk
U1. <same structure plus missing evidence>

Validation
- <check>: NOT RUN | PASS | FAIL | BLOCKED | NOT APPLICABLE

Checks Not Performed
- <check and reason>

Suggested Next Steps
1. <ordered action>
```

Omit empty finding sections only after explicitly stating `None identified`. Do not paste long raw build output; summarize it and preserve the relevant error/warning text only.

In assessment mode, finish with `PARTIALLY VALIDATED` unless concrete required findings justify `CHANGES REQUIRED`, the assessment is blocked, or the project is unsupported. Never return `READY` from static assessment alone.

### Step 5: Prepare validation commands

In `validate` mode, choose commands based on repository documentation and operating system. Prefer repository wrappers.

First show a plan similar to:

```text
Proposed commands (execute repository code):
1. <wrapper version command>
2. <clean verification/build command>
3. <project-specific test/integration command, if any>
4. <jdeps command against produced artifacts, if applicable>
5. <jdeprscan command against produced artifacts, if applicable>

These commands may execute build plugins and download dependencies from configured repositories.
Proceed?
```

Use project-documented commands when present. Otherwise propose reasonable defaults:

- Maven on Unix-like systems: `./mvnw clean verify`
- Maven on Windows: `.\mvnw.cmd clean verify`
- Gradle on Unix-like systems: `./gradlew clean build --no-daemon`
- Gradle on Windows: `.\gradlew.bat clean build --no-daemon`

If no wrapper exists, report that fact and propose system Maven/Gradle only if installed. Do not install it.

Run `jdeps --jdk-internals` and `jdeprscan --for-removal` only after successful artifact creation and only against relevant application artifacts. Determine classpaths/modules from the build rather than inventing them. State that static analysis can miss reflective access.

### Step 6: Execute approved validation

After explicit approval:

1. Confirm `java -version` and `javac -version` identify JDK 21.
2. Run only the approved commands in order.
3. Stop or ask before materially expanding beyond the approved command set.
4. Capture exit status and concise relevant output.
5. Distinguish migration failures from pre-existing test failures, missing credentials/network access, and sandbox resource limits.
6. Do not fix failures automatically.

If the installed JDK is not 21, return `BLOCKED` for runtime validation while still providing the static assessment.

### Step 7: Produce the final validation report

Reprint the full report, not only the delta. Update each validation check with `PASS`, `FAIL`, `BLOCKED`, or `NOT APPLICABLE` and include evidence.

Return:

- `READY` only when the baseline's readiness gate is fully met.
- `CHANGES REQUIRED` when concrete required changes remain.
- `BLOCKED` when a prerequisite prevents meaningful assessment/validation.
- `PARTIALLY VALIDATED` when static analysis succeeded but required runtime evidence is incomplete.

Always summarize remaining high-risk or unresolved items, even when all executed commands pass.

## Guidelines

- Prefer evidence over generic Java migration advice.
- Keep required and advisory items distinct.
- Explain recommendations in plain language.
- Do not claim transitive dependency compatibility from direct dependency versions alone.
- Do not assume a successful compile proves runtime, integration, performance, security, charset, or deployment compatibility.
- Do not recommend adopting Java 21 language features as part of compatibility migration; that is optional modernization.
- Preserve human accountability: generated guidance and results require developer/Java SME review.
