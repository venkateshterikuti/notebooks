---
name: java-17-to-21
description: Assess and validate a Maven or Gradle Java 17 to 21 migration
---

# Java 17 to 21 migration

Assess the current local Java repository and produce an evidence-based Java 17 to Java 21 migration report. Follow the `java-17-to-21-baseline` instruction supplied by this pack.

Invocation arguments: {args}

Interpret an argument containing `validate` as validation mode. Otherwise use assessment mode. Arguments may also identify a local repository path, but never inspect a different repository unless the user explicitly supplied it.

## Usage

```text
/java-17-to-21
/java-17-to-21 validate
```

- Assessment mode performs static, read-only discovery and does not execute repository code.
- Validation mode performs the same assessment, proposes exact commands, and executes them only after explicit approval.

## Operating contract

- Use `glob_files`, `grep`, and `read_file` for static discovery and evidence collection.
- Use `bash` only when shell execution is necessary. Follow the active Anteroom safety policy even for read-only shell commands.
- Use `ask_user` to obtain explicit approval before Maven, Gradle, wrapper, application, test, `jdeps`, or `jdeprscan` execution.
- Treat wrappers and build scripts as repository code. Approval for one command does not authorize materially different commands.
- Do not use `write_file`, `edit_file`, `save_memory`, or `run_agent` in this skill.
- Do not install software, change dependencies, commit, push, deploy, or update external systems.
- Do not send repository content to an online service merely to perform the assessment.
- Redact secrets and sensitive configuration values.
- Mark facts that cannot be resolved as `UNRESOLVED`; never guess effective values or compatibility.

## Workflow

### Step 1: Establish repository context

1. Use the current workspace unless an explicit local path was provided.
2. Discover Java source, build files, wrapper files, CI files, Dockerfiles, and nested modules with `glob_files`.
3. Use `grep` to locate Java-version properties, compiler settings, toolchains, plugins, dependencies, JVM flags, and risk patterns.
4. Use `read_file` to inspect relevant files and capture file-and-line evidence.
5. Detect Maven, Gradle, mixed builds, nested independent builds, and unsupported build systems.
6. Return `UNSUPPORTED` when no applicable Java repository exists or the repository is outside the baseline scope.

Do not assume the installed Java version is the repository's configured source, target, release, or runtime version.

### Step 2: Discover Java and build configuration

Inspect all applicable modules.

For Maven, inspect:

- POM modules, parents, profiles, properties, plugin management, and dependency management
- `maven.compiler.release`, compiler `<release>`, `<source>`, `<target>`, and `java.version`
- Maven Wrapper and `.mvn/` configuration
- Toolchains, compiler, Surefire, Failsafe, Javadoc, JaCoCo, packaging, and framework plugins

For Gradle, inspect:

- Groovy/Kotlin build and settings files, `gradle.properties`, version catalogs, and convention plugins
- Java toolchains, source/target compatibility, `options.release`, and test launchers
- Root, subproject, `buildSrc`, and included-build overrides
- Wrapper distribution version and relevant plugin versions

For delivery and runtime, inspect:

- CI configuration, Dockerfiles, buildpacks, deployment manifests, and startup scripts
- `.java-version`, `.sdkmanrc`, `.tool-versions`, toolchain declarations, and committed IDE settings
- JVM environment variables and flags without exposing secret values

Report contradictions separately. Do not run Maven or Gradle to calculate effective configuration in assessment mode.

### Step 3: Analyze migration areas

Create findings for:

1. Java compiler/release configuration
2. Maven/Gradle wrapper and build plugins
3. Dependencies, BOMs, platforms, and frameworks
4. Annotation processors and code generation
5. Tests, coverage, bytecode, and agent tooling
6. Internal JDK APIs, reflection, and module-access flags
7. Removed/deprecated APIs and obsolete JVM-option indicators
8. Charset, locale, serialization, crypto/TLS, JNI/native, and behavioral risks
9. CI, container, packaging, and deployment-runtime alignment
10. Missing validation coverage

Classify each finding according to the pack baseline:

- `REQUIRED`: repository or approved command evidence proves a migration change is needed.
- `ADVISORY`: a resilience or maintainability recommendation not proven to block Java 21.
- `UNRESOLVED`: available evidence cannot establish compatibility.

Do not label a dependency incompatible solely because it is old. Exact version recommendations require an authoritative dated source or approved organizational baseline.

Treat `--add-opens` as evidence of reflective access to a strongly encapsulated package. It does not by itself prove use of an internal JDK API or that the flag can be removed.

Treat an unavailable JDK 21 or build tool as a blocked validation prerequisite, not as a required repository change.

### Step 4: Produce the assessment report

Start with a concise executive summary, followed by detailed evidence:

```text
Java 17 -> 21 Migration Assessment

Executive Summary
Overall status: <CHANGES REQUIRED | BLOCKED | PARTIALLY VALIDATED | UNSUPPORTED>
Mode: <assessment | validation>
Top required changes: <count and short list>
Validation blockers: <count and short list>
Highest risks: <short list>

Repository: <path>
Build: <Maven | Gradle | Mixed> <wrapper version or unresolved>
Configured Java: <value(s) with evidence>
Installed Java: <value, not checked, or unavailable>

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

For an empty finding section, write `None identified`. Do not paste long raw output; preserve only concise, relevant errors or warnings.

Static assessment cannot return `READY`. Use `CHANGES REQUIRED` when concrete required changes exist, `BLOCKED` when meaningful static assessment cannot proceed, `UNSUPPORTED` when outside scope, and otherwise `PARTIALLY VALIDATED`.

### Step 5: Prepare validation commands

In validation mode, select commands from repository documentation and the operating system. Prefer wrappers. Show every proposed command, working directory, purpose, side effects, and dependency-download possibility before execution.

```text
Proposed commands (execute repository code):
1. <JDK/build-tool version command>
2. <clean verification/build command>
3. <project-specific tests or startup command>
4. <jdeps command against built artifacts, if applicable>
5. <jdeprscan command against built artifacts, if applicable>

These commands may execute build plugins and download dependencies.
Proceed?
```

Use `ask_user` and wait for explicit approval. Do not treat the invocation word `validate` as approval to execute commands.

Default proposals, only when repository documentation does not define commands:

- Maven Unix-like: `./mvnw clean verify`
- Maven Windows: `.\mvnw.cmd clean verify`
- Gradle Unix-like: `./gradlew clean build --no-daemon`
- Gradle Windows: `.\gradlew.bat clean build --no-daemon`

If no wrapper exists, report it and propose system Maven or Gradle only when already installed. Never install it.

Run `jdeps --jdk-internals` and `jdeprscan --for-removal` only after successful artifact creation and only against relevant artifacts. Do not invent a classpath or module path.

### Step 6: Execute approved validation

After approval:

1. Confirm `java -version` and `javac -version` report JDK 21.
2. Run only the approved commands with `bash`, in order.
3. Stop and ask before expanding the command set.
4. Record command, working directory, exit status, and concise relevant output.
5. Distinguish migration failures from pre-existing failures, missing credentials/network, unavailable tooling, and sandbox limits.
6. Do not fix failures automatically.

If JDK 21 is unavailable, mark JDK-dependent checks `BLOCKED`. Continue reporting static repository findings; do not create a required-change finding merely because the local environment lacks Java.

### Step 7: Produce the final validation report

Reprint the full report and update each validation item to `PASS`, `FAIL`, `BLOCKED`, or `NOT APPLICABLE` with evidence.

- Return `READY` only when the complete baseline readiness gate passes.
- Return `CHANGES REQUIRED` when concrete repository changes remain.
- Return `BLOCKED` when a prerequisite prevents meaningful assessment or validation and no more specific status applies.
- Return `PARTIALLY VALIDATED` when static analysis succeeded but required runtime evidence remains incomplete.

Always report remaining unresolved and high-risk items, even when all approved commands pass.

## Guidelines

- Prefer evidence over generic migration advice.
- Keep required, advisory, unresolved, and validation-blocker items distinct.
- Explain recommendations in plain language.
- Do not claim transitive dependency compatibility from direct versions alone.
- Do not assume successful compilation proves runtime, integration, performance, security, charset, or deployment compatibility.
- Do not include optional Java 21 language-feature adoption in compatibility scope.
- Require developer and Java SME review of generated guidance and results.
