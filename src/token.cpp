#include "token.h"
#include "token_engine.h"

namespace ISQCrit {

void Token::Release() {
    if (refcount_ == 0) return; // defensive; should not happen
    if (--refcount_ == 0 && engine_) {
        engine_->TokenExpire(this);
    }
}

bool Token::DoNotifyExecuted() {
    if (!engine_) return false;
    return engine_->UpdateCriticality(this);
}

} // namespace ISQCrit
